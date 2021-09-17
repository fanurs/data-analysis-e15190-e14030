# Data Analysis: E15190-E14030
**Useful external links**
- [Experiment homepage](https://groups.nscl.msu.edu/hira/15190-14030/index.htm)
- [Run log (WMU)](http://neutronstar.physics.wmich.edu/runlog/index.php?op=list)
- [Overview of experimental runs](https://groups.nscl.msu.edu/hira/fanurs/progress/20210615.html)

**Background**

This repository primarily focuses on neutron wall analysis. Instead of starting from the raw event files (binary files from the DAQ), we treat the "unpacked" ROOT files generated using [Daniele's analysis framework](https://github.com/nscl-hira/E15190-Unified-Analysis-Framework) (actually we use Kuan's version, which is not on GitHub) as inputs, and from there we write our own analysis codes. In the case where calibration parameters are not good enough or some minor mistakes were found in the framework, we always try *not* to modify the framework. Instead, we would like to "freeze" the framework, and correct for any imperfection or errors in the output ROOT files within this repository. Unless there is something that framework does not preserve in the ROOT files, e.g. missing events, then only we will go back and try to debug the framework.

This decision was made after considering the fact that all major authors of the framework had left the group. It is simply safer and more efficient to keep it as "legacy code", and build on top of the ROOT files it generated. Fortunately, most of the calibrations have already been done. While some are found to be off later on, most are already good enough for a simple "first-order" analysis.

**Table of contents**
1. [Installation](#1-installation)
1. [Structure of the repository](#2-structure-of-the-repository)
1. [Unit tests](#3-unit-tests)

## 1. Installation
If you are not familiar with conda, more detailed instructions can be found at [`doc/installation.md`](doc/installation.md).
1. Git clone the repository:
```console
git clone https://github.com/Fanurs/data-analysis-e15190-e14030.git
```
2. Install (if haven't) and activate `conda`.
3. Build the conda environment. This step is *time-consuming* (~hours). Use a screen session whenever possible.
```console
cd data-analysis-e15190-e14030/
./build.py
```
4. Activate the conda environment:
```console
conda activate ./env_e15190/
```
5. To quickly check if the conda environment has been activated, inspect the environment variable `$PROJECT_DIR`:
```console
echo $PROJECT_DIR
```

## 2. Structure of the repository
This repository is written mainly in Python 3.8 and C++20 (`-std=c++2a` in GCC 9 and earlier).
- **`e15190/`**: Core Python source codes for doing calibration and analysis
- **`database/`**: For storing all the calibration parameters, cache files (e.g. `.h5` files), images, ROOT files, Mako templates, etc.
- **`local/`**: Miscellaneous local configuration files, scripts, etc. C++ source codes also go into here, mainly to separate them from the Python source codes in `e15190/`.
- **`scripts/`**: Here are all the calibration scripts and batch scripts (for parallel computing).
- **`tests/`**: Unit tests.
- **`env_e15190/`**: Where the conda environment is built.
- **`doc/`**: Documentation.
- **`environment.yml`**: Configuration file for setting up the conda environment.
- **`build.py`**: Installation script. To build the conda environment as well as modifying a few other things, e.g. environment variables, terminal commands, etc.

## 3. Unit tests
We are using the [`pytest`](https://docs.pytest.org/) framework. To test everything, simply activate the conda environment, go do the project directory and type:
```console
pytest
```
To test a specify file or directory, e.g. `tests/utilities/test_timer.py`:
```console
pytest tests/utilities/test_timer.py
```

**When to do unit testing?**
* Right after installation, run all tests to see if things are working correctly.
* After changing any codes, run all tests to check in case things that were previously working are now broken, a.k.a. regression testing. [<span style="color: red;">Important!</span>]
* You are also encouraged to write and run tests *while* developing the source code. This is often considered as the best practice, though sometimes a little too tedious.
* Any other scenarios where you think there might be a chance to break things, i.e. server updates, conda environment updates, git merge with other collaborators, etc.
