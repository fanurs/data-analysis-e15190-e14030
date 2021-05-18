"""This is a module that manages all local files, i.e. files that are not being committed to GitHub.

Local files are not being committed to GitHub because they may contain contents that are user-dependent.
Here is a list of non-exhaustive examples:
- Local paths or directories
- Local libraries or environments
- Sensitive information like passwords that should never be committed to git
- Source codes or data that are export control or intellectual properties of other parties
- Experimental scripts that are often modified

While users can, of course, maintain as many local files as they want, some local files are essential to use the modules.
These local files often have their specific formats, which this module is written to check them as well.
"""
import json
import pathlib

from .. import PROJECT_DIR

class LocalPathsManager:
    def __init__(self, check=True):
        self.LOCAL_PATHS_JSON_PATH = pathlib.Path(PROJECT_DIR, 'database', 'local_paths.json')
        with open(self.LOCAL_PATHS_JSON_PATH, 'r') as file:
            self.content = json.load(file)
        self.required_keys = {
            'daniele_root_files_dir': str,
        }

        if check:
            self.check()

    def check(self):
        correct = True
        for key, typ in self.required_keys.items():
            if key not in self.content:
                print(f'Entry for "{key}" is not found!')
                correct = False
            elif not isinstance(self.content[key], typ):
                print(f'Value type for "{key}" is incorrect. It should be {str(typ)}.')
                correct = False
        return correct

_local_paths_manager = LocalPathsManager()

def check_local_paths_json(*args, **kwargs):
    return _local_paths_manager.check(*args, **kwargs)

def get_local_path(key):
    return _local_paths_manager.content[key]