import os
from pathlib import Path
import shutil

import e15190

e15190.DATABASE_DIR = Path(os.environ['PROJECT_DIR']) / '.database_test'
os.environ['DATABASE_DIR'] = str(e15190.DATABASE_DIR)

shutil.rmtree(e15190.DATABASE_DIR, ignore_errors=True)
e15190.DATABASE_DIR.mkdir(parents=True, exist_ok=True)

def copy_from_default_database(rel_path, not_found_ok=False):
    """
    Parameters
    ----------
    rel_path : str or pathlib.Path
        File path relative to the database directory.
    not_found_ok : bool, default False
        If True, raises FileNotFoundError if the file is not found.
    
    Returns
    -------
    copy_success : bool
        True if the file was copied, False otherwise.
    """
    src = e15190._DEFAULT_DATABASE_DIR / rel_path
    if not src.is_file():
        if not_found_ok:
            return False
        raise FileNotFoundError(f'File not found: "{str(src)}"')
    dst = e15190.DATABASE_DIR / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    return True

# files needed for most tests
copy_from_default_database('runlog/elog_runs_filtered.h5')
copy_from_default_database('runlog/run_batches.csv')

from . import generate_sample_roots as gsr
if not gsr.Sample1.root_exists():
    gsr.Sample1.generate()
