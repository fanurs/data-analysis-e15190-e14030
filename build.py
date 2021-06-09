import sys
if not (sys.version_info[0] >= 3 and sys.version_info[1] >= 7):
    raise Exception('Requires at least Python 3.7 to use this repository.')

import glob
import os
import pathlib
import subprocess

ENVIRONMENT_NAME = 'env_e15190'

def main():
    check_conda_activated()

    # create a conda environment locally
    subprocess.run(f'conda env create --prefix ./{ENVIRONMENT_NAME} -f environment.yml', shell=True)

    # add packages in this project/repository to (editable) site-packages of this conda environment
    site_pkgs_path = glob.glob(f'./{ENVIRONMENT_NAME}/lib/python*/site-packages')[0]
    path = pathlib.Path(site_pkgs_path, 'conda.pth').resolve()
    project_dir = pathlib.Path(__file__).parent.resolve()
    with open(path, 'w') as file:
        file.write(str(project_dir) + '\n')

    # add custom terminal commands from scripts/ to $ENV/bin
    script_paths = []
    for path in glob.iglob('./scripts/*'):
        path = pathlib.Path(path)
        if path.is_file() and os.access(path, os.X_OK) and path.stem == path.name:
            script_paths.append(path.resolve())
    for script_path in script_paths:
        symbol_path = pathlib.Path(ENVIRONMENT_NAME, 'bin', script_path.name)
        try:
            symbol_path.symlink_to(script_path)
        except FileExistsError:
            print(f'FileExistsError: "{script_path.name}" already exists under "{ENVIRONMENT_NAME}/bin/"')

    print('Done!')

def check_conda_activated():
    which_conda = subprocess.run('which conda', shell=True, capture_output=True, text=True)
    path1 = pathlib.Path(which_conda.stdout.strip())
    path1 = pathlib.Path(path1.parent.parent, 'conda-meta')
    path2 = pathlib.Path(sys.prefix, 'conda-meta')
    if not (path1.is_dir() and path2.is_dir()):
        raise Exception('Fail to detect conda environment. Please make sure you have activated conda.')

if __name__ == '__main__':
    main()