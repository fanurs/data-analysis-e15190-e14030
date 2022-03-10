import inspect
import json
import pathlib
import urllib.request

from cryptography.fernet import Fernet
import pandas as pd
import pymysql
pymysql.install_as_MySQLdb() # WMU uses MySQL

from e15190 import PROJECT_DIR

MYSQL_DOWNLOAD_PATH = 'database/runlog/downloads/mysql_database.h5'
"""Local path where MySQL database is downloaded to as HDF5 file."""

ELOG_DOWNLOAD_PATH = 'database/runlog/downloads/elog.html'
"""Local path where ELOG is downloaded to as HTML file."""

KEY_PATH = 'database/key_for_all.pub'
"""Local path where secret key is stored.

This key is used to decrypt the credentials needed to connect to the MySQL. It
should not be committed to the repository.
"""

# The URL of the e-log, hosted at Western Michigan University (WMU)
ELOG_URL = 'http://neutronstar.physics.wmich.edu/runlog/index.php?op=list'

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
    def __init__(self, auto_connect=False, key_path=None, verbose=True):
        """Constructor for :py:class:`MySqlDownloader`.

        Parameters
        ----------
        auto_connect : bool, default False
            Whether to automatically connect to the MySQL database. Login
            credentials are needed to connect to the database.
        key_path : str, default None
            File path to the key used to decrypt the login credential at
            ``$PROJEC_DIR/database/runlog/mysql_login_credential.json``.

            !! This key should never be committed to the repository. !!
            
            Ask the owner of this repository for the key.
        verbose : bool, default True
            Whether to print the progress of connecting to the MySQL database.
            This setting has less priority than individual class functions, i.e.
            if other functions have explicitly set a verbose value, then this
            global setting will be ignored.
        """
        self.connection = None
        """``pymysql.Connection`` object.

        See more at
        https://pymysql.readthedocs.io/en/latest/modules/connections.html
        """

        self.cursor = None
        """``pymysql.Cursor`` object.

        See more at
        https://pymysql.readthedocs.io/en/latest/modules/cursors.html
        """

        self.verbose = verbose
        """The global verbose setting. Default is ``True``."""

        if auto_connect:
            self.connect(key_path=key_path)
    
    def __enter__(self):
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

    def connect(self, key_path=None, verbose=None):
        """Establish connection to the MySQL database.

        Upon successful connection, ``self.connection`` is set to the connection
        object and ``self.cursor`` is set to the cursor object. Fetch functions
        of ``self.cursor`` are decorated to return lists of lists, including
        ``fetchall``, ``fetchone`` and ``fetchmany``.
        
        Parameters
        ----------
        key_path : str, default None
            File path to the key used to decrypt the login credential at
            ``$PROJEC_DIR/database/runlog/mysql_login_credential.json``.
            If ``None``, the key is read from the file at
            ``$PROJEC_DIR/database/key_for_all.pub``. This is also the key used
            for all other purposes in this project.

            !! This key should never be committed to the repository. !!

            Ask the owner of this repository for the key.
        verbose : bool, default None
            Whether to print the progress of connecting to the MySQL database.
            If ``None``, the global setting is used.

        Raises
        ------
        FileNotFoundError
            If the key file is not found.
        """
        verbose = self.verbose if verbose is None else verbose

        key_path = PROJECT_DIR / KEY_PATH if key_path is None else pathlib.Path(key_path)
        if not key_path.is_file():
            raise FileNotFoundError(inspect.cleandoc(
                f'''Key is not found at
                "{str(key_path)}"
                If the key has been provided to you, please check if the path is
                correct. Otherwise, contact the owner of this repository for
                more help.
                '''))
        with open(key_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line[0] == '#' or len(line) == 0:
                continue
            secret_key = line

        # get credential
        credential_path = PROJECT_DIR / 'database/runlog/mysql_login_credential.json'
        with open(credential_path, 'r') as f:
            credential = json.load(f)
        
        # decrypt credential
        for key, val in credential.items():
            fernet_key = Fernet(secret_key)
            credential[key] = fernet_key.decrypt(val.encode('utf-8')).decode('utf-8')

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
        return pd.read_sql(f'SELECT * FROM {table_name}', self.connection)

    def download(
        self,
        download_path=None,
        table_names=None,
        auto_disconnect=False,
        verbose=True):
        """Convert tables into pandas dataframes and save into an HDF file

        Parameters
        ----------
        download_path : str, default None
            File path to the HDF file. If ``None``, the file is saved at
            ``$PROJECT_DIR/database/runlog/downloads/mysql_database.h5``.
        auto_disconnect : bool, default False
            Whether to automatically disconnect from the MySQL database after
            all tables have been downloaded.
        table_names : list of str, default None
            List of table names to download. If ``None``, all tables are
            downloaded.
        verbose : bool, default True
            Whether to print the progress of downloading. If ``None``, the
            global setting is used.
        """
        verbose = self.verbose if verbose is None else verbose

        print('Attempting to download run log from WMU MySQL database...')
        self.cursor.execute('SHOW TABLES')
        table_names = self.cursor.fetchall() if table_names is None else table_names
        download_path = PROJECT_DIR / MYSQL_DOWNLOAD_PATH if download_path is None else pathlib.Path(download_path)

        download_path.parent.mkdir(parents=True, exist_ok=True)
        if download_path.is_file():
            resp = input(inspect.cleandoc(
                f'''HDF file already exists at
                "{PROJECT_DIR / MYSQL_DOWNLOAD_PATH}".
                Do you want to re-download from WMU MySQL database? (y/n)
                This will overwrite the existing file.
                > '''
                ))
            if not resp.lower().strip == 'y':
                print('No re-download will be performed.')
        else:
            resp = 'y'

        if resp.lower().strip() == 'y':
            download_path.unlink(missing_ok=True)
            with pd.HDFStore(download_path, 'w') as file:
                for table_name in table_names:
                    if verbose:
                        print(f'> Converting and saving {table_name}... ', end='', flush=True)
                    df = pd.read_sql(f'SELECT * FROM {table_name}', self.connection)
                    file.append(table_name, df)
                    print('Done!')
            print(f'All tables have been saved to\n"{str(download_path)}"')

        if auto_disconnect:
            self.disconnect()

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

class ElogDownloader:
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
            ``$PROJECT_DIR/database/runlog/downloads/elog.html``.
        verbose : bool, default True
            Whether to print the progress of downloading.
        timeout : int, default 3
            Timeout in seconds for the request.
        read_nbytes : int, default None
            Number of bytes to read from the webpage. If ``None``, the entire
            webpage is read, decoded, and saved. This is useful for testing.
        """
        download_path = PROJECT_DIR / ELOG_DOWNLOAD_PATH if download_path is None else pathlib.Path(download_path)
        if verbose:
            print(f'Attempting to download web content from\n"{ELOG_URL}"... ', end='', flush=True)
        web_request = urllib.request.urlopen(ELOG_URL, timeout=timeout)
        web_content = web_request.read(read_nbytes)
        (PROJECT_DIR / ELOG_DOWNLOAD_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(download_path, 'wb') as file:
            file.write(web_content)
        if verbose:
            print()
            print('Done!')
