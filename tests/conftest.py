import os
from pathlib import Path
import shutil

import e15190

OLD_DATABASE_DIR = e15190.DATABASE_DIR
e15190.DATABASE_DIR = Path(os.environ['PROJECT_DIR']) / '.database_test'
os.environ['DATABASE_DIR'] = str(e15190.DATABASE_DIR)

os.system(f'rm -rf {str(e15190.DATABASE_DIR)}/*')
e15190.DATABASE_DIR.mkdir(parents=True, exist_ok=True)

def copy(rel_path):
    dst = e15190.DATABASE_DIR / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(OLD_DATABASE_DIR / rel_path, dst)
copy('runlog/mysql_login_credential.json')