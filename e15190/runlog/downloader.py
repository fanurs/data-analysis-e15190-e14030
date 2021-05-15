import json
import pathlib

from cryptography.fernet import Fernet
import pandas as pd
import pymysql

pymysql.install_as_MySQLdb() # WMU uses MySQL

PROJECT_DIR = pathlib.Path(__file__).parent.parent.parent.resolve()

class Downloader:
    def __init__(self, auto_connect=True):
        if auto_connect:
            self.connect()

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

    def connect(self):
        # get public key (same for all scripts in this repository; not committed to github)
        key_path = pathlib.Path(PROJECT_DIR, 'database', 'key_for_all.pub')
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
        
        self.cursor.fetchall = Downloader.decorate(self.cursor.fetchall)
        self.cursor.fetchmany = Downloader.decorate(self.cursor.fetchmany)
        self.cursor.fetchone = Downloader.decorate(self.cursor.fetchone)
    
    def download(self, auto_disconnect=True):
        self.cursor.execute('SHOW TABLES')
        table_names = self.cursor.fetchall()

        # convert tables into pandas dataframes and save into an HDF file
        download_path = pathlib.Path(PROJECT_DIR, 'database', 'runlog', 'wmu_mysql_database.h5')
        download_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.HDFStore(download_path, 'w') as file:
            for table_name in table_names:
                print(f'> Converting and saving {table_name}...')
                df = pd.read_sql(f'SELECT * FROM {table_name}', self.connection)
                file.append(table_name, df)

        if auto_disconnect:
            self.disconnect()
        
    def disconnect(self):
        self.connection.close()