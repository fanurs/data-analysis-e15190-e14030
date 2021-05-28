import sys
if not (sys.version_info[0] >= 3 and sys.version_info[1] >= 7):
    raise Exception('Requires at least Python 3.7 to use this repository.')

import glob
import pathlib
import subprocess

ENVIRONMENT_NAME = 'env_e15190'

def main():
    check_conda_activated()

    # create a conda environment locally
    subprocess.run(f'conda env create --prefix ./{ENVIRONMENT_NAME} -f simple_env.yml', shell=True)

    # add packages in this repository to (editable) site-packages of this conda environment
    site_pkgs_path = glob.glob(f'./{ENVIRONMENT_NAME}/lib/python*/site-packages')[0]
    path = pathlib.Path(site_pkgs_path, 'conda.pth').resolve()
    subprocess.run(f'echo `pwd` > {str(path)}', shell=True)

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