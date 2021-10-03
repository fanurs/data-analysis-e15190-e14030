"""This is a module that manages all local files.

Local files are not being committed to GitHub because they may contain contents
that are user-dependent or confidential. Here is a list of non-exhaustive examples:

- Local paths or directories
- Local libraries or environments
- Sensitive information like passwords that should *never* be committed to git
- Source codes or data that are export controlled or intellectual properties of other parties
- Experimental scripts that are frequently modified and not ready for release

This module is written to checked and manage a few of these local files that are
crucial to running repository.
"""
import json
import pathlib

from e15190 import PROJECT_DIR

class LocalPathsManager:
    def __init__(self, check=False):
        self.LOCAL_PATHS_JSON_PATH = PROJECT_DIR / 'database/local_paths.json'
        """``pathlib.Path`` : ``PROJECT_DIR / 'database/local_paths.json``"""

        self.required_keys = {
            'daniele_root_files_dir': str,
        }
        """dict : Essential keys and their types.

        Keys that must be present in the :py:attr:`LOCAL_PATHS_JSON_PATH` file.
        ::
            {
                'daniele_root_files_dir': str,
            }
        """

        if not self.LOCAL_PATHS_JSON_PATH.is_file():
            self.create_template()

        with open(self.LOCAL_PATHS_JSON_PATH, 'r') as file:
            self.content = json.load(file)

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

    def create_template(self):
        with open(self.LOCAL_PATHS_JSON_PATH, 'w') as file:
            template = {key: None for key in self.required_keys.keys()}
            print(template)
            json.dump(template, file, indent=4)

_local_paths_manager = LocalPathsManager()

def check_local_paths_json(*args, **kwargs):
    return _local_paths_manager.check(*args, **kwargs)

def get_local_path(key):
    return _local_paths_manager.content[key]