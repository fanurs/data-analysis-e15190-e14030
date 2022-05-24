from inspect import cleandoc
import os
from pathlib import Path

from cryptography.fernet import Fernet

def _get_key_from_path(key_path):
    key_path = os.path.expandvars(key_path)
    key_path = Path(key_path)

    if not key_path.is_file():
        return None

    with open(key_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line[0] == '#' or len(line) == 0:
                continue
            return line

def _get_key_from_env(env_var):
    if env_var in os.environ:
        return os.environ[env_var]

def get_key(
    key_path='$PROJECT_DIR/.key_for_all.pub',
    env_var='KEY_E15190',
    force_key_path=False,
    force_env_var=False,
):
    if force_key_path:
        return _get_key_from_path(key_path)
    if force_env_var:
        return _get_key_from_env(env_var)

    key = _get_key_from_path(key_path) or _get_key_from_env(env_var)
    if key is None:
        raise FileNotFoundError(cleandoc(
            f'''Key can neither be found at
            "{str(key_path)}"
            nor as the environment variable, ${env_var}.
            If the key has been provided to you, please check if the path is set
            up correctly. Otherwise, contact the owner of this repository for
            more.
            '''
        ))
    return key

def encrypt(message, key):
    ferney_key = Fernet(key)
    return ferney_key.encrypt(message.encode('utf-8')).decode('utf-8')

def decrypt(encrypted_value, key):
    ferney_key = Fernet(key)
    return ferney_key.decrypt(encrypted_value.encode('utf-8')).decode('utf-8')
