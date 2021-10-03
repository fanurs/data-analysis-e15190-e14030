.. _installation-detailed-version:
.. highlight:: console
.. toctree::
   :maxdepth: 2
   :caption: Contents:

*******************************
Installation (detailed version)
*******************************


1. Git clone
============

Go to the directory where you want to install this repository and type
::
   user@server:~$ git clone https://github.com/Fanurs/data-analysis-e15190-e14030.git

You should now see a directory named ``data-analysis-e15190-e14030``. We shall use ``$PROJECT_DIR`` to denote it from now on.


2. Install conda
================

This repository uses conda environment. Conda is a cross-platform package management system and environment management system. It was initially created to manage Python libraries, but now it can be used to install libraries (and compilers) for C++, Fortran, Java, Javascripts, etc. So this is an ideal tool especially when working on a remote server which you don't have administrator privilege. In fact, we are going to install some complicated softwares like ROOT, which takes only one command. If you do not have conda installed yet, you may try one of the following two options:

- Download Anaconda or Miniconda (recommended because light-weighted) by yourself. See https://docs.anaconda.com/anaconda/install/linux/ or https://docs.conda.io/en/latest/miniconda.html for more instructions, or check out the script at ``$PROJECT_DIR/local/autoinstall-Miniconda-3.x.sh``. You should *not* install conda inside this repository. Install it to somewhere else with sufficient disk space (~3 GB) (I used ``/projects/hira``). We are using Python 3.8 for this project.

- Use conda pre-installed on Fishtank/HPCC. You can view them by typing ``module avail`` on the terminal. Please choose the latest version; at the time of writing, the latest version can be loaded with ``module load anaconda/python3.7``. While this option is easier than to install conda by yourself, you run the risk that something may break when the server administrators decide to modify the installation in the future.

In the rest of this writing, I will use ``$CONDA_DIR`` to denote the directory where you have installed conda.


3. Customize ``~/.condarc`` (optional)
======================================

This step is definitely not mandatory, but I find a few settings are very useful to have, so you might as well want to just add them in since the very beginning. First, create a file ``~/.condarc`` if you don't already have it. Open it with your favorite text editor. Here, I provide just three settings that I believe should be the most useful ones for one to get started.
::
   env_prompt: '($(basename) {default_env})) '
   channels:
      - defaults
      - conda-forge
   pkgs_dirs:
      - /this/is/just/an/example/mnt/directorywithatleast10GB/.conda/pkgs

If this is your first time using conda, you probably don't know what these mean yet. Just copy and paste what I have here to get started. The only thing you need to modify is, of course, the directory for ``pkgs_dirs``. This is a directory where later conda will put all the downloaded ``.tar.gz``, ``.zip`` files and so on. Usually you don't need to interact with this directory manually, so a not-so-popular place with sufficient storage (~10 GB) would be ideal.

See https://conda.io/projects/conda/en/latest/user-guide/configuration/use-condarc.html for more ``.condarc`` options.


4. Activate base conda environment
==================================

This can be done by entering
::
   user@server:~$ source "$CONDA_DIR/bin/activate"

Type ``which python`` to check if you are using the python inside this conda directory. Something like ``$CONDA_DIR/bin/python`` should show up. You may want to save this command to your ``.bashrc`` or ``.bash_profile`` for convenience. See more at https://linuxize.com/post/bashrc-vs-bash-profile/.


5. Building conda environment for this repository
=================================================

You are ready to install this repository. This step is very time-consuming (~hours), especially if this is your first time building any conda environment. Starting from the second time, conda will start looking for some cached installations (``pkgs_dirs`` in your ``~/.condarc``). So make sure you have a very stable internet connection, or even better, use a [screen session](https://linuxize.com/post/how-to-use-linux-screen/) (strongly encouraged).
::
   (base) user@server:~$ cd data-analysis-e15190-e14030
   (base) user@server:data-analysis-e15190-e14030$ python build.py

This script will first create a local conda environment at ``$PROJECT_DIR/env_e15190/``, and install all the packages specified in ``$PROJECT_DIR/environment.yml``. It will also add all the local packages like ``$PROJECT_DIR/e15190`` as site packages, i.e. once you have loaded this conda environment, you can simply write ``import e15190`` in your Python script without causing ``ImportNotFoundError``.

6. Done
=======

That's everything. Next time, you just have to activate the conda environment by typing something like
::
   (base) user@server:data-analysis-e15190-e14030$ conda activate ./env_e15190
   (env_e15190) user@server:data-analysis-e15190-e14030$ 

and start working.
