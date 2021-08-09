# Data Analysis: E15190-E14030

Experiment homepage: https://groups.nscl.msu.edu/hira/15190-14030/index.htm


## Installation
1. Go to the directory where you want to install this repository and type:
```console
user@server:~$ git clone https://github.com/Fanurs/data-analysis-e15190-e14030.git
```
You should now see a directory named `data-analysis-e15190-e14030`. We shall use `$PROJECT_DIR` to denote it from now on.

2. This repository uses conda environment. If you do not have conda installed yet, you may try:
    - Download Anaconda or Miniconda (recommended because light-weighted) by yourself. See https://docs.anaconda.com/anaconda/install/linux/ for more instructions, or check out the script at `$PROJECT_DIR/local/autoinstall-Miniconda-3.x.sh`. You should *not* install conda inside this repository. Install it to somewhere else with sufficient disk space.
    - Use conda pre-installed on Fishtank/HPCC. You can view them by typing `module avail` on the terminal. Please choose the latest version; at the time of writing, the latest version can be loaded with `module load anaconda/python3.7`. While this option is easier than to install conda by yourself, you run the risk that something may break when the server administrators decide to modify the installation in the future.

3. Now, activate your base conda environment. This can usually be done by entering
```console
user@server:~$ source $CONDA_DIR/bin/activate
```
Type `which python` to check if you are using the python inside this conda directory. Something like `$CONDA_DIR/bin/python` should show up. You may want to save this command to your `.bashrc` or `.bash_profile` for convenience. See more at https://linuxize.com/post/bashrc-vs-bash-profile/.

4. You are ready to install this repository.
```console
(base) user@server:~$ cd data-analysis-e15190-e14030
(base) user@server:data-analysis-e15190-e14030$ python build.py
```
This script will first create a local conda environment at `$PROJECT_DIR/env_e15190/`, and install all the packages specified in `$PROJECT_DIR/environment.yml`. It will also add all the local packages like `$PROJECT_DIR/e15190` as site packages, i.e. once you have loaded this conda environment, you can simply write `import e15190` in your Python script without causing `ImportNotFoundError`.

5. That's basically it. Next time, you just have to activate the conda environment by typing something like
```console
(base) user@server:data-analysis-e15190-e14030$ conda activate ./env_e15190
(env_e15190) user@server:data-analysis-e15190-e14030$ 
```
and start working. Good luck!
