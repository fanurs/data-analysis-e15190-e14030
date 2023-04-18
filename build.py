#!/usr/bin/env python
import sys
if not (sys.version_info[0] >= 3 and sys.version_info[1] >= 7):
    raise Exception('Requires at least Python 3.7 to use this repository.')

import glob
import inspect
import os
import pathlib
import subprocess

ENVIRONMENT_NAME = 'env_e15190'

def main():
    check_conda_activated()

    # create a conda environment locally
    subprocess.run(['conda', 'env', 'create', '--prefix', f'./{ENVIRONMENT_NAME}', '-f', 'environment.yml'])

    # add packages in this project/repository to (editable) site-packages of this conda environment
    site_pkgs_path = glob.glob(f'./{ENVIRONMENT_NAME}/lib/python*/site-packages')[0]
    path = pathlib.Path(site_pkgs_path, 'conda.pth').resolve()
    project_dir = pathlib.Path(__file__).parent.resolve()
    with open(path, 'w') as file:
        file.write(str(project_dir) + '\n')

    # add custom terminal commands from scripts/ to $ENV/bin
    script_paths = []
    for path in glob.iglob('./local/bin/*'):
        path = pathlib.Path(path)
        if path.is_file() and os.access(path, os.X_OK) and path.stem == path.name:
            script_paths.append(path.resolve())
    for script_path in script_paths:
        symbol_path = pathlib.Path(ENVIRONMENT_NAME, 'bin', script_path.name)
        if symbol_path.is_symlink():
            symbol_path.unlink()
        symbol_path.symlink_to(script_path)

    # add activation and deactivation scripts
    env_vars_content = {
        'activate': inspect.cleandoc(f'''
            #!/bin/bash
            if [ ! -z "$PROJECT_DIR" ]; then
                export OLD_PROJECT_DIR=$PROJECT_DIR
            fi
            export PROJECT_DIR="{str(pathlib.Path('.').resolve())}"
        '''),
        'deactivate': inspect.cleandoc(f'''
            #!/bin/bash
            if [ ! -z "$OLD_PROJECT_DIR" ]; then
                PROJECT_DIR=$OLD_PROJECT_DIR
                unset OLD_PROJECT_DIR
            else
                unset PROJECT_DIR
            fi
        '''),
    }
    src_parent_dir = pathlib.Path('./local/etc/conda')
    des_parent_dir = pathlib.Path(ENVIRONMENT_NAME, 'etc/conda')
    base_fname = 'env_vars.sh'
    for name in ['activate', 'deactivate']:
        src_subdir = pathlib.Path(src_parent_dir, f'{name}.d')
        src_subdir.mkdir(parents=True, exist_ok=True)
        src_path = pathlib.Path(src_subdir, f'{name}-{base_fname}')
        src_path.touch(exist_ok=True)
        with open(src_path, 'w') as file:
            file.write(env_vars_content[name] + '\n')

        des_subdir = pathlib.Path(des_parent_dir, f'{name}.d')
        des_subdir.mkdir(parents=True, exist_ok=True)
        des_path = pathlib.Path(des_subdir, f'{name}-{base_fname}')
        if des_path.is_symlink():
            des_path.unlink()
        des_path.resolve().symlink_to(src_path.resolve())
    
    print('Done!')

def check_conda_activated():
    which_conda = subprocess.run(['which', 'conda'], capture_output=True, text=True)
    path1 = pathlib.Path(which_conda.stdout.strip())
    path1 = pathlib.Path(path1.parent.parent, 'conda-meta')
    path2 = pathlib.Path(sys.prefix, 'conda-meta')
    if not (path1.is_dir() and path2.is_dir()):
        raise Exception('Fail to detect conda environment. Please make sure you have activated conda.')

if __name__ == '__main__':
    main()
