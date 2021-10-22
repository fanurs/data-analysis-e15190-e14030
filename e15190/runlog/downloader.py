import inspect
import json
import pathlib
import urllib.request

from cryptography.fernet import Fernet
import pandas as pd
import pymysql
pymysql.install_as_MySQLdb() # WMU uses MySQL

from e15190 import PROJECT_DIR

MYSQL_DOWNLOAD_PATH = PROJECT_DIR / 'database/runlog/downloads/mysql_database.h5'
ELOG_DOWNLOAD_PATH = PROJECT_DIR / 'database/runlog/downloads/elog.html'
KEY_PATH = PROJECT_DIR / 'database/key_for_all.pub'
ELOG_URL = 'http://neutronstar.physics.wmich.edu/runlog/index.php?op=list'

class MySqlDownloader:
    """This class downloads the MySQL database from WMU.
    
    Due to some in-house security measures, the MySQL database is stored at
    Western Michigan University. To analyze the data, we download the database
    from WMU and store it locally at NSCL/FRIB's server. All tables are
    downloaded as pandas dataframes, and stored in an HDF files. This allows
    quicker access to the data and more complicated analysis.
    """
    def __init__(self, auto_connect=True, key_path=None):
        """Constructor for :py:class:`MySqlDownloader`.

        Parameters
        ----------
        auto_connect : bool, default True
            Whether to automatically connect to the MySQL database. Login
            credentials are needed to connect to the database.
        key_path : str, default None
            File path to the key used to decrypt the login credential at
            ``$PROJEC_DIR/database/runlog/mysql_login_credential.json``.

            !! This key should never be committed to the repository. !!
            
            Ask the owner of this repository for the key.
        """
        if auto_connect:
            self.connect(key_path=key_path)

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

    def connect(self, key_path=None, verbose=True):
        """Establish connection to the MySQL database.
        
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
        verbose : bool, default True
            Whether to print the progress of connecting to the MySQL database.

        Raises
        ------
        FileNotFoundError
            If the key file is not found.
        """
        key_path = KEY_PATH if key_path is None else key_path
        if not key_path.is_file():
            raise FileNotFoundError(inspect.cleandoc(
                f'''Key is not found at
                "{str(key_path)}"
                If the key has been provided to you, please check if the path is
                correct.  Otherwise, contact the owner of this repository for
                more help.
                '''))
        with open(key_path, 'r') as f:
            secret_key = f.readline().encode('utf-8')

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
        
        self.cursor.fetchall = MySqlDownloader.decorate(self.cursor.fetchall)
        self.cursor.fetchmany = MySqlDownloader.decorate(self.cursor.fetchmany)
        self.cursor.fetchone = MySqlDownloader.decorate(self.cursor.fetchone)

    def get_all_table_names(self):
        """Returns all table names in the MySQL database.
        """
        self.cursor.execute('SHOW TABLES')
        return self.cursor.fetchall()
    
    def download(
        self,
        download_path=None,
        table_names=None,
        auto_disconnect=True,
        verbose=True):
        """Convert tables into pandas dataframes and save into an HDF file

        Parameters
        ----------
        download_path : str, default None
            File path to the HDF file. If ``None``, the file is saved at
            ``$PROJECT_DIR/database/runlog/downloads/mysql_database.h5``.
        auto_disconnect : bool, default True
            Whether to automatically disconnect from the MySQL database after
            all tables have been downloaded.
        table_names : list of str, default None
            List of table names to download. If ``None``, all tables are
            downloaded.
        verbose : bool, default True
            Whether to print the progress of downloading.
        """
        print('Attempting to download run log from WMU MySQL database...')
        self.cursor.execute('SHOW TABLES')
        table_names = self.cursor.fetchall() if table_names is None else table_names
        download_path = MYSQL_DOWNLOAD_PATH if download_path is None else pathlib.Path(download_path)

        download_path.parent.mkdir(parents=True, exist_ok=True)
        if download_path.is_file():
            resp = input(inspect.cleandoc(
                f'''HDF file already exists at
                "{MYSQL_DOWNLOAD_PATH}".
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

    def disconnect(self, verbose=True):
        """Disconnect from the MySQL database.

        Parameters
        ----------
        verbose : bool, default True
            Whether to print the progress of disconnecting.
        """
        self.connection.close()
        if verbose:
            print('Connection to MySQL database at WMU has been closed.')

class ElogDownloader:
    def __init__(self):
        """Downloads the runlog from the webpage.
        """
        pass

    def download(self, download_path=None, verbose=True):
        """Downloads the runlog from the webpage.

        Parameters
        ----------
        download_path : str, default None
            File path to the Excel file. If ``None``, the file is saved at
            ``$PROJECT_DIR/database/runlog/downloads/elog.html``.
        verbose : bool, default True
            Whether to print the progress of downloading.
        """
        download_path = ELOG_DOWNLOAD_PATH if download_path is None else pathlib.Path(download_path)
        if verbose:
            print(f'Attempting to download web content from\n"{ELOG_URL}"... ', end='', flush=True)
        web_content = urllib.request.urlopen(ELOG_URL).read()
        ELOG_DOWNLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(download_path, 'wb') as file:
            file.write(web_content)
        if verbose:
            print()
            print('Done!')
