import inspect
import json
import os
from pathlib import Path
import sqlite3
import urllib.request
import warnings

import pandas as pd
import pymysql

from e15190.utilities import key_manager

class ElogDownloader:
    DOWNLOAD_PATH = '$DATABASE_DIR/runlog/downloads/elog.html'
    URL = 'http://neutronstar.physics.wmich.edu/runlog/index.php?op=list'

    def __init__(self):
        """Initialize the ElogDownloader.

        The Elog is viewable at
        http://neutronstar.physics.wmich.edu/runlog/index.php?op=list
        """
        pass

    def download(
        self,
        download_path=None,
        verbose=True,
        timeout=3,
        read_nbytes=None,
    ):
        """Downloads the runlog from the webpage.

        Parameters
        ----------
        download_path : str, default None
            File path to the Excel file. If ``None``, the file is saved at
            ``$DATABASE_DIR/runlog/downloads/elog.html``.
        verbose : bool, default True
            Whether to print the progress of downloading.
        timeout : int, default 3
            Timeout in seconds for the request.
        read_nbytes : int, default None
            Number of bytes to read from the webpage. If ``None``, the entire
            webpage is read, decoded, and saved. This is useful for testing.

        Returns
        -------
        download_path : pathlib.Path
            Path to the downloaded HTML file.
        """
        if download_path is None:
            download_path = os.path.expandvars(self.DOWNLOAD_PATH)
        download_path = Path(download_path)
        download_path.parent.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f'Attempting to download web content from\n"{self.URL}"... ', end='', flush=True)
        web_request = urllib.request.urlopen(self.URL, timeout=timeout)
        web_content = web_request.read(read_nbytes)
        with open(download_path, 'wb') as file:
            file.write(web_content)
        if verbose:
            print()
            print('Done!')
        return download_path

class MySqlDownloader:
    """This class downloads the MySQL database from WMU.
    
    Due to some in-house security measures, the MySQL database is stored at
    Western Michigan University. To analyze the data, we download the database
    from WMU and store it locally at NSCL/FRIB's server. All tables are
    downloaded as pandas dataframes, and stored in an HDF files. This allows
    quicker access to the data and more complicated analysis.

    It is encouraged to use the ``with`` statement when interacting with this
    class. Here is an example:

    >>> from e15190.runlog import downloader
    >>> with downloader.MySqlDownloader(auto_connect=True) as dl:
    >>>     df = dl.get_table('runtarget')
    """
    CREDENTIAL_PATH = '$DATABASE_DIR/runlog/mysql_login_credential.json'
    DOWNLOAD_PATH = '$DATABASE_DIR/runlog/downloads/mysql_database.db'

    def __init__(self, auto_connect=False, verbose=True):
        """Constructor for :py:class:`MySqlDownloader`.

        Parameters
        ----------
        auto_connect : bool, default False
            Whether to automatically connect to the MySQL database. Login
            credentials are needed to connect to the database. If you are using
            context manager, this parameter is irrelevant --- connection is
            always being established.
        verbose : bool, default True
            Whether to print the progress of connecting to the MySQL database.
            This setting has less priority than individual class functions, i.e.
            if other functions have explicitly set a verbose value, then this
            global setting will be ignored.
        """
        pymysql.install_as_MySQLdb() # WMU uses MySQL
        self.connection = None
        self.cursor = None
        self.verbose = verbose

        if auto_connect:
            self.connect()
    
    def __enter__(self):
        if self.connection is None:
            self.connect()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Disconnect from the MySQL database upon exiting the context.
        """
        self.disconnect()

    @staticmethod
    def decorate(func_that_returns_tuples_of_tuples):
        """Converts a function that returns tuples of tuples into a function
        that returns tuples of lists.
        
        This is primarily used to decorate
        :py:meth:`MySqlDownloader.cursor.fetchall`.

        Parameters
        ----------
        func_that_returns_tuples_of_tuples : function
            Function that returns tuples of tuples, e.g. ``cursor.fetchall()``.
        
        Returns
        -------
        deco_func : function
            Function that returns a list of lists.
        """
        f = func_that_returns_tuples_of_tuples
        def inner(*args, **kwargs):
            arr = [list(ele) if isinstance(ele, tuple) else ele for ele in f(*args, **kwargs)]

            # reduce the dimensionality if inner arrays are single-element array
            if all([isinstance(ele, list) and len(ele) == 1 for ele in arr]):
                arr = [ele[0] for ele in arr]
            return arr
        return inner

    def connect(self, verbose=None):
        """Establish connection to the MySQL database.

        Upon successful connection, ``self.connection`` is set to the connection
        object and ``self.cursor`` is set to the cursor object. Fetch functions
        of ``self.cursor`` are decorated to return lists of lists, including
        ``fetchall``, ``fetchone`` and ``fetchmany``.
        
        Parameters
        ----------
        verbose : bool, default None
            Whether to print the progress of connecting to the MySQL database.
            If ``None``, the global setting is used.

        Raises
        ------
        FileNotFoundError
            If the key file is not found.
        """
        verbose = self.verbose if verbose is None else verbose
        with open(os.path.expandvars(self.CREDENTIAL_PATH), 'r') as file:
            credential = json.load(file)
        
        # decrypt credential
        secret_key = key_manager.get_key()
        for key, val in credential.items():
            credential[key] = key_manager.decrypt(val, secret_key)

        # establish connection to the database
        self.connection = pymysql.connect(**credential, db='hiramodules')
        del credential
        self.cursor = self.connection.cursor()
        if verbose:
            print('Connection to MySQL database at WMU has been established.', flush=True)
        
        # decorate fetch functions so that they return lists of lists
        self.cursor.fetchall = MySqlDownloader.decorate(self.cursor.fetchall)
        self.cursor.fetchmany = MySqlDownloader.decorate(self.cursor.fetchmany)
        self.cursor.fetchone = MySqlDownloader.decorate(self.cursor.fetchone)

    def get_all_table_names(self):
        """Returns all table names in the MySQL database.

        Returns
        -------
        table_names : list
            List of all table names in the MySQL database.
        """
        self.cursor.execute('SHOW TABLES')
        return self.cursor.fetchall()
    
    def get_table(self, table_name):
        """Returns the table as a pandas dataframe.

        Parameters
        ----------
        table_name : str
            Name of the table to be downloaded.

        Returns
        -------
        table : pandas.DataFrame
            Table as a pandas dataframe.
        
        Raises
        ------
        ValueError
            If the table is not found.
        """
        all_table_names = self.get_all_table_names()
        if table_name not in all_table_names:
            raise ValueError(f'Table "{table_name}" is not found in the database.')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return pd.read_sql(f'SELECT * FROM {table_name}', self.connection)

    def download(
        self,
        download_path=None,
        table_names=None,
        auto_disconnect=False,
        verbose=True):
        """Convert tables into pandas dataframes and save into an SQLite3 file

        Parameters
        ----------
        download_path : str, default None
            File path to the HDF file. If ``None``, the file is saved at
            ``$DATABASE_DIR/runlog/downloads/mysql_database.db``.
        auto_disconnect : bool, default False
            Whether to automatically disconnect from the MySQL database after
            all tables have been downloaded.
        table_names : list of str, default None
            List of table names to download. If ``None``, all tables are
            downloaded.
        verbose : bool, default True
            Whether to print the progress of downloading. If ``None``, the
            global setting is used.

        Returns
        -------
        download_path : pathlib.Path
            File path to the SQLite3 file.
        """
        verbose = self.verbose if verbose is None else verbose

        print('Attempting to download run log from WMU MySQL database...')
        self.cursor.execute('SHOW TABLES')
        table_names = self.cursor.fetchall() if table_names is None else table_names
        if download_path is None:
            download_path = os.path.expandvars(self.DOWNLOAD_PATH)
        download_path = Path(download_path)

        download_path.parent.mkdir(parents=True, exist_ok=True)
        if download_path.is_file():
            resp = input(inspect.cleandoc(
                f'''SQLite3 file already exists at
                "{str(download_path)}".
                Do you want to re-download from WMU MySQL database? (y/n)
                This will overwrite the existing file.
                > '''
                ))
            if not resp.lower().strip() == 'y':
                print('No re-download will be performed.')
        else:
            resp = 'y'

        if resp.lower().strip() == 'y':
            download_path.unlink(missing_ok=True)
            with sqlite3.connect(download_path) as sqlite_conn:
                for table_name in table_names:
                    if verbose:
                        print(f'Downloading table "{table_name}"... ', end='', flush=True)
                    df = self.get_table(table_name)
                    df.to_sql(table_name, sqlite_conn, if_exists='replace')
                    if verbose:
                        print('Done!', flush=True)
            print(f'All tables have been saved to\n"{str(download_path)}"')

        if auto_disconnect:
            self.disconnect()
        
        return download_path

    def disconnect(self, verbose=None):
        """Disconnect from the MySQL database.

        Parameters
        ----------
        verbose : bool, default None
            Whether to print the progress of disconnecting. If ``None``, the
            global setting is used.
        """
        verbose = self.verbose if verbose is None else verbose
        if self.connection is not None:
            self.connection.close()
            self.connection = None
            self.cursor = None
            if verbose:
                print('Connection to MySQL database at WMU has been closed.')
        else:
            if verbose:
                print('No connection found. Nothing to disconnect.')

if __name__ == '__main__': # pragma: no cover
    print('''
    What do you want to download?
    \t1) Elog data from the web
    \t2) Scalers data from the MySQL database at WMU
    ''')
    resp = input('(1/2) > ')
    if resp == '1':
        elog_downloader = ElogDownloader()
        elog_downloader.download()
        exit()
    if resp == '2':
        with MySqlDownloader() as mysql_downloader:
            mysql_downloader.download()
        exit()
    print('Invalid input.')
