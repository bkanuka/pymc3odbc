"""ODBC trace backend

Store and retrieve sampling values in ODBC database.

Database format
---------------
For each variable, a table is created with the following format:

 recid (INT), draw (INT), chain (INT),  v0 (FLOAT), v1 (FLOAT), v2 (FLOAT) ...

The variable column names are extended to reflect additional dimensions.
For example, a variable with the shape (2, 2) would be stored as

 key (INT), draw (INT), chain (INT),  v0_0 (FLOAT), v0_1 (FLOAT), v1_0 (FLOAT) ...

The key is autoincremented each time a new row is added to the table.
The chain column denotes the chain index and starts at 0.
"""
import numpy as np
import pyodbc

from pymc3.backends import base, ndarray
from pymc3.backends import tracetab as ttab

TEMPLATES = {
    'table':            ('DROP TABLE IF EXISTS [{table}]'
                         'CREATE TABLE [{table}] '
                         '(recid INTEGER NOT NULL PRIMARY KEY IDENTITY, '
                         'draw INTEGER, chain INT, '
                         '{value_cols})'),
    'insert':           ('INSERT INTO [{table}] '
                         '(draw, chain, {value_cols}) '
                         'VALUES (?, ?, {values})'),
    'max_draw':         ('SELECT MAX(draw) FROM [{table}] '
                         'WHERE chain = ?'),
    'draw_count':       ('SELECT COUNT(*) FROM [{table}] '
                         'WHERE chain = ?'),
    # Named placeholders are used in the selection templates because
    # some values occur more than once in the same template.
    'select':           ('SELECT * FROM [{table}] '
                         'WHERE (chain = {chain})'),
    'select_burn':      ('SELECT * FROM [{table}] '
                         'WHERE (chain = {chain}) AND (draw > {burn})'),
    'select_thin':      ('SELECT * FROM [{table}] '
                         'WHERE (chain = {chain}) AND '
                         '(draw - (SELECT draw FROM [{table}] '
                         'WHERE chain = {chain} '
                         'ORDER BY draw LIMIT 1)) % {thin} = 0'),
    'select_burn_thin': ('SELECT * FROM [{table}] '
                         'WHERE (chain = {chain}) AND (draw > {burn}) '
                         'AND (draw - (SELECT draw FROM [{table}] '
                         'WHERE (chain = {chain}) AND (draw > {burn}) '
                         'ORDER BY draw LIMIT 1)) % {thin} = 0'),
    'select_point':     ('SELECT * FROM [{table}] '
                         'WHERE (chain = {chain}) AND (draw = {draw})'),
}


class ODBC(base.BaseTrace):
    """ODBC trace object

    Parameters
    ----------
    con_str : str
        ODBC Connection string including database

    model : Model
        If None, the model is taken from the `with` context.
    vars : list of variables
        Sampling values will be stored for these variables. If None,
        `model.unobserved_RVs` is used.
    test_point : dict
        use different test point that might be with changed variables shapes
    """

    def __init__(self, con_str, model=None, vars=None, test_point=None):
        self.con_str = con_str
        # Parse the con_str to get db name
        self.name = _get_db_name(con_str)
        super(ODBC, self).__init__(self.name, model, vars, test_point)
        self._var_cols = {}
        self.var_inserts = {}  # varname -> insert statement
        self.draw_idx = 0
        self._is_setup = False
        self._len = None

        self.db = _ODBCDB(self.con_str)
        # Inserting sampling information is queued to avoid locks
        # caused by hitting the database with transactions each
        # iteration.
        self._queue = {varname: [] for varname in self.varnames}
        self._queue_limit = 5000

    # Sampling methods

    def setup(self, draws, chain):
        """Perform chain-specific setup.

        Parameters
        ----------
        draws : int
            Expected number of draws
        chain : int
            Chain number
        """
        self.db.connect()
        self.chain = chain

        if self._is_setup:
            self.draw_idx = self._get_max_draw(chain) + 1
            self._len = None
        else:  # Table has not been created.
            self._var_cols = {varname: ttab.create_flat_names('v', shape)
                              for varname, shape in self.var_shapes.items()}
            self._create_table()
            self._is_setup = True
        self._create_insert_queries()

    def _create_table(self):
        template = TEMPLATES['table']
        with self.db.con:
            for varname, var_cols in self._var_cols.items():
                if np.issubdtype(self.var_dtypes[varname], np.int):
                    dtype = 'INT'
                else:
                    dtype = 'FLOAT'
                colnames = ', '.join([v + ' ' + dtype for v in var_cols])
                statement = template.format(table=varname,
                                            value_cols=colnames)
                self.db.cursor.execute(statement)

    def _create_insert_queries(self):
        template = TEMPLATES['insert']
        for varname, var_cols in self._var_cols.items():
            # Create insert statement for each variable.
            var_str = ', '.join(var_cols)
            placeholders = ', '.join(['?'] * len(var_cols))
            statement = template.format(table=varname,
                                        value_cols=var_str,
                                        values=placeholders)
            self.var_inserts[varname] = statement

    def record(self, point):
        """Record results of a sampling iteration.

        Parameters
        ----------
        point : dict
            Values mapped to variable names
        """
        for varname, value in zip(self.varnames, self.fn(point)):
            values = (self.draw_idx, self.chain) + tuple(np.ravel(value))
            self._queue[varname].append(values)

        if len(self._queue[self.varnames[0]]) > self._queue_limit:
            self._execute_queue()
        self.draw_idx += 1

    def _execute_queue(self):
        with self.db.con:
            for varname in self.varnames:
                if not self._queue[varname]:
                    continue
                self.db.cursor.executemany(self.var_inserts[varname],
                                           self._queue[varname])
                self._queue[varname] = []

    def close(self):
        self._execute_queue()
        self.db.close()

    # Selection methods

    def __len__(self):
        if not self._is_setup:
            return 0
        if self._len is None:
            self._len = self._get_number_draws()
        return self._len

    def _get_number_draws(self):
        self.db.connect()
        statement = TEMPLATES['draw_count'].format(table=self.varnames[0])
        self.db.cursor.execute(statement, (self.chain,))
        counts = self.db.cursor.fetchall()[0][0]
        if counts is None:
            return 0
        else:
            return counts

    def _get_max_draw(self, chain):
        self.db.connect()
        statement = TEMPLATES['max_draw'].format(table=self.varnames[0])
        self.db.cursor.execute(statement, (chain,))
        counts = self.db.cursor.fetchall()[0][0]
        if counts is None:
            return 0
        else:
            return counts

    def get_values(self, varname, burn=0, thin=1):
        """Get values from trace.

        Parameters
        ----------
        varname : str
        burn : int
        thin : int

        Returns
        -------
        A NumPy array
        """
        if burn is None:
            burn = 0
        if thin is None:
            thin = 1

        if burn < 0:
            burn = max(0, len(self) + burn)
        if thin < 1:
            raise ValueError('Only positive thin values are supported '
                             'in ODBC backend.')
        varname = str(varname)

        statement_args = {'table': varname,
                          'chain': self.chain}
        if burn == 0 and thin == 1:
            action = 'select'
        elif thin == 1:
            action = 'select_burn'
            statement_args['burn'] = burn - 1
        elif burn == 0:
            action = 'select_thin'
            statement_args['thin'] = thin
        else:
            action = 'select_burn_thin'
            statement_args['burn'] = burn - 1
            statement_args['thin'] = thin

        self.db.connect()
        shape = (-1,) + self.var_shapes[varname]
        statement = TEMPLATES[action].format(**statement_args)
        self.db.cursor.execute(statement)
        values = _rows_to_ndarray(self.db.cursor)
        return values.reshape(shape)

    def _slice(self, idx):
        if idx.stop is not None:
            raise ValueError('Stop value in slice not supported.')
        return ndarray._slice_as_ndarray(self, idx)

    def point(self, idx):
        """Return dictionary of point values at `idx` for current chain
        with variables names as keys.
        """
        idx = int(idx)
        if idx < 0:
            idx = self._get_max_draw(self.chain) + idx + 1
        statement = TEMPLATES['select_point']
        self.db.connect()
        var_values = {}
        statement_args = {'chain': self.chain, 'draw': idx}
        for varname in self.varnames:
            self.db.cursor.execute(statement.format(table=varname),
                                   statement_args)
            values = _rows_to_ndarray(self.db.cursor)
            var_values[varname] = values.reshape(self.var_shapes[varname])
        return var_values


class _ODBCDB(object):
    def __init__(self, con_str):
        self.con_str = con_str
        self.con = None
        self.cursor = None
        self.connected = False

    def connect(self):
        if self.connected:
            return
        self.con = pyodbc.connect(self.con_str)
        self.connected = True
        self.cursor = self.con.cursor()

    def close(self):
        if not self.connected:
            return
        self.con.commit()
        self.cursor.close()
        self.con.close()
        self.connected = False


def load(con_str, model=None):
    """Load ODBC database.

    Parameters
    ----------
    con_str : str
        ODBC Connection string including database
    model : Model
        If None, the model is taken from the `with` context.

    Returns
    -------
    A MultiTrace instance
    """
    db = _ODBCDB(con_str)
    db.connect()
    name = _get_db_name(con_str)
    varnames = _get_table_list(db.cursor)
    if len(varnames) == 0:
        raise ValueError(('Can not get variable list for database'
                          '`{}`'.format(name)))
    chains = _get_chain_list(db.cursor, varnames[0])

    print(chains)
    straces = []
    for chain in chains:
        strace = ODBC(con_str, model=model)
        strace.chain = chain
        strace._var_cols = {varname: ttab.create_flat_names('v', shape)
                            for varname, shape in strace.var_shapes.items()}
        strace._is_setup = True
        strace.db = db  # Share the db with all traces.
        straces.append(strace)
    return base.MultiTrace(straces)


def _get_db_name(con_str):
    con_dict = dict([[d.strip() for d in c.split('=')] for c in con_str.split(';') if len(c) > 1])
    con_dict = dict((k.lower(), v) for k, v in con_dict.items())
    return con_dict['database']

def _get_table_list(cursor):
    """Return a list of table names in the current database."""
    # Modified from Django. Skips the sqlite_sequence system table used
    # for autoincrement key generation.
    cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES \
                   WHERE TABLE_TYPE='BASE TABLE' \
                   ORDER BY TABLE_NAME")
    tables = [row[0] for row in cursor.fetchall()]
    print(tables)
    return tables


def _get_var_strs(cursor, varname):
    cursor.execute('SELECT * FROM [{}]'.format(varname))
    col_names = (col_descr[0] for col_descr in cursor.description)
    return [name for name in col_names if name.startswith('v')]


def _get_chain_list(cursor, varname):
    """Return a list of sorted chains for `varname`."""
    cursor.execute('SELECT DISTINCT chain FROM [{}]'.format(varname))
    chains = [chain[0] for chain in cursor.fetchall()]
    chains.sort()
    return chains


def _rows_to_ndarray(cursor):
    """Convert SQL row to NDArray."""
    return np.squeeze(np.array([row[3:] for row in cursor.fetchall()]))