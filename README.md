# Data Analysis: E15190-E14030

Experiment homepage: https://groups.nscl.msu.edu/hira/15190-14030/index.htm


## Installation
1. Go to the directory where you want to install this repository and type:<br>
`git clone https://github.com/Fanurs/data-analysis-e15190-e14030.git`<br>
You should now see a directory named `data-analysis-e15190-e149030`. We shall use `$PROJECT_DIR` to denote it from now on.

1. This repository uses conda environment. If you do not have conda installed yet, you may try:
    - Download Anaconda or Miniconda (recommended because light-weighted) by yourself. See https://docs.anaconda.com/anaconda/install/linux/ for more instructions, or check out the script at `$PROJECT_DIR/local/autoinstall-Miniconda-3.x.sh`. You should *not* install conda inside this repository. Install it to somewhere else with sufficient disk space.
    - Use conda pre-installed on Fishtank. You can view them by typing `module avail` on the terminal. Please choose the latest version; at the time of writing, the latest one can be invoked with `module load anaconda/python3.7`. Nonetheless, you run this risk that something may change when the server administrators decide to modify the installation.

1. Now, activate your base conda environment. This can usually be done by entering<br>
`source $CONDA_DIR/bin/activate`<br>
Type `which python` to check if you are using the python inside this conda directory. Something like `$CONDA_DIR/bin/python` should show up.

1. You are ready to install this repository. Make sure you are inside `$PROJECT_DIR`. Then do<br>
`python build.py`<br>
This script will first create a local conda environment at `$PROJECT_DIR/env_e15190/` for this repository according to `$PROJECT_DIR/environment.yml`. Then it will add all the custom packages like `$PROJECT_DIR/e15190` as site packages.

1. That's basically it. Next time, you just have to activate the conda environment by typing something like<br>
`conda activate $PROJECT_DIR/env_e15190`<br>
and start working. Good luck!
