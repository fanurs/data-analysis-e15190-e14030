import json
import pathlib

from e15190 import PROJECT_DIR

def set_matplotlib_style(mpl):
    path = pathlib.Path(PROJECT_DIR, 'database/utilities/styles/matplotlib.json')
    with open(path, 'r') as file:
        content = json.load(file)
    mpl.rcParams.update(content)
