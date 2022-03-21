import os
from pathlib import Path
import shutil

import e15190

e15190.DATABASE_DIR = Path(os.environ['PROJECT_DIR']) / '.database_test'
os.environ['DATABASE_DIR'] = str(e15190.DATABASE_DIR)

shutil.rmtree(e15190.DATABASE_DIR, ignore_errors=True)
e15190.DATABASE_DIR.mkdir(parents=True, exist_ok=True)

def copy_from_default_database(rel_path):
    src = e15190._DEFAULT_DATABASE_DIR / rel_path
    if not src.is_file():
        return False
    dst = e15190.DATABASE_DIR / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    return True