import pathlib
import os

_DEFAULT_PROJECT_DIR = pathlib.Path(__file__).parent.parent
if 'PROJECT_DIR' in os.environ:
    PROJECT_DIR = pathlib.Path(os.environ['PROJECT_DIR'])
else:
    PROJECT_DIR = _DEFAULT_PROJECT_DIR
    os.environ['PROJECT_DIR'] = str(PROJECT_DIR)

_DEFAULT_DATABASE_DIR = PROJECT_DIR / 'database'
if 'DATABASE_DIR' in os.environ:
    DATABASE_DIR = pathlib.Path(os.environ['DATABASE_DIR'])
else:
    DATABASE_DIR = _DEFAULT_DATABASE_DIR
    os.environ['DATABASE_DIR'] = str(DATABASE_DIR)