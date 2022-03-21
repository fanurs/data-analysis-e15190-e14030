import pathlib
import os

PROJECT_DIR = pathlib.Path(os.environ['PROJECT_DIR'])
DATABASE_DIR = PROJECT_DIR / 'database'
os.environ['DATABASE_DIR'] = str(DATABASE_DIR)