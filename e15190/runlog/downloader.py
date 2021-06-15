import inspect
import json
import pathlib
import urllib.request

from cryptography.fernet import Fernet
import pandas as pd
import pymysql

pymysql.install_as_MySQLdb() # WMU uses MySQL

from .. import PROJECT_DIR
from . import MYSQL_DOWNLOAD_PATH, ELOG_DOWNLOAD_PATH
ELOG_URL = 'http://neutronstar.physics.wmich.edu/runlog/index.php?op=list'

class MySqlDownloader:
    def __init__(self, auto_connect=True, public_key_path=None):
        if auto_connect:
            self.connect(public_key_path=public_key_path)

    @staticmethod
    def decorate(func_that_returns_tuples_of_tuples):
        f = func_that_returns_tuples_of_tuples
        def inner(*args, **kwargs):
            arr = [list(ele) if isinstance(ele, tuple) else ele for ele in f(*args, **kwargs)]

            # reduce the dimensionality if inner arrays are single-element array
            if all([isinstance(ele, list) and len(ele) == 1 for ele in arr]):
                arr = [ele[0] for ele in arr]
            return arr
        return inner

    def connect(self, public_key_path=None):
        # get public key (same for all scripts in this repository; not committed to github)
        key_path = pathlib.Path(PROJECT_DIR, 'database', 'key_for_all.pub') if public_key_path is None else public_key_path
        if not key_path.is_file():
            raise Exception(inspect.cleandoc(
                f'''Public key is not found at
                "{str(key_path)}"
                If the key has been provided to you, please check if the path is correct.
                Otherwise, contact the owner of this repository for more help.
                '''))
        with open(key_path, 'r') as f:
            pub_key = f.readline().encode('utf-8')

        # get credential
        credential_path = pathlib.Path(PROJECT_DIR, 'database', 'runlog', 'mysql_login_credential.json')
        with open(credential_path, 'r') as f:
            credential = json.load(f)
        
        # decrypt credential
        for key, val in credential.items():
            credential[key] = Fernet(pub_key).decrypt(val.encode('utf-8')).decode('utf-8')

        # establish connection to the database
        self.connection = pymysql.connect(**credential, db='hiramodules')
        del credential
        self.cursor = self.connection.cursor()
        print('Connection to MySQL database at WMU has been established.', flush=True)
        
        self.cursor.fetchall = MySqlDownloader.decorate(self.cursor.fetchall)
        self.cursor.fetchmany = MySqlDownloader.decorate(self.cursor.fetchmany)
        self.cursor.fetchone = MySqlDownloader.decorate(self.cursor.fetchone)
    
    def download(self, auto_disconnect=True, verbose=True):
        """Convert tables into pandas dataframes and save into an HDF file
        """
        print('Attempting to download run log from WMU MySQL database...')
        self.cursor.execute('SHOW TABLES')
        table_names = self.cursor.fetchall()

        MYSQL_DOWNLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
        if MYSQL_DOWNLOAD_PATH.is_file():
            resp = input(inspect.cleandoc(
                f'''HDF file already exists at
                "{MYSQL_DOWNLOAD_PATH}".
                Do you want to re-download from WMU MySQL database? This will overwrite the existing file.
                (y/n)
                > '''
                ))
            if not resp.lower().strip == 'y':
                print('No re-download will be performed.')
        else:
            resp = 'y'

        if resp.lower().strip() == 'y':
            MYSQL_DOWNLOAD_PATH.unlink(missing_ok=True)
            with pd.HDFStore(MYSQL_DOWNLOAD_PATH, 'w') as file:
                for table_name in table_names:
                    if verbose:
                        print(f'> Converting and saving {table_name}... ', end='', flush=True)
                    df = pd.read_sql(f'SELECT * FROM {table_name}', self.connection)
                    file.append(table_name, df)
                    print('Done!')
            print(f'All tables have been saved to\n"{str(MYSQL_DOWNLOAD_PATH)}"')

        if auto_disconnect:
            self.disconnect()
        
    def disconnect(self):
        self.connection.close()

class ElogDownloader:
    def __init__(self):
        pass

    def download(self):
        print(f'Attempting to download web content from\n"{ELOG_URL}"... ', end='', flush=True)
        web_content = urllib.request.urlopen(ELOG_URL).read()
        ELOG_DOWNLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ELOG_DOWNLOAD_PATH, 'wb') as file:
            file.write(web_content)
        print('Done!')