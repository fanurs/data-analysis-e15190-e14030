# Data Analysis: E15190-E14030
**Useful links**
- [:closed_book: Documentation](https://fanurs.github.io/data-analysis-e15190-e14030/build/html/index.html)
- [:radioactive: Experiment homepage](https://groups.nscl.msu.edu/hira/15190-14030/index.htm)
- [:memo: Run log (WMU)](http://neutronstar.physics.wmich.edu/runlog/index.php?op=list)
- [:chart_with_upwards_trend:	 Plotly: Experimental runs](https://groups.nscl.msu.edu/hira/fanurs/progress/20210615.html)

**Background**

This repository primarily focuses on neutron wall analysis. Instead of starting from the raw event files (binary files from the DAQ), we treat the "unpacked" ROOT files generated using [Daniele's analysis framework](https://github.com/nscl-hira/E15190-Unified-Analysis-Framework) (currently using branch `zhu`) as inputs, and from there we write our own analysis codes. In the case where calibration parameters are not good enough or some minor mistakes were found in the framework, we always try *not* to modify the framework. Instead, we would like to "freeze" the framework, and correct for any imperfection or errors in the output ROOT files within this repository. Unless there is something that framework does not preserve in the ROOT files, e.g. missing events, then only we will go back and try to debug the framework (currently WMU is working on this).

This decision was made after considering the fact that all major authors of the framework had left the group. It is simply safer and more efficient to keep it as "legacy code", and build on top of the ROOT files it generated. Fortunately, most of the calibrations have already been done. While some are found to be off later on, most are already good enough for a simple "first-order" analysis.

**Table of contents**
1. [Installation](#1-installation)
1. [Structure of the repository](#2-structure-of-the-repository)
1. [Testing framework](#3-testing-framework)
1. [PEP 8 style guide](#4-pep-8-style-guide)

## 1. Installation
If you are not familiar with conda, more detailed instructions can be found at [Installation (detailed version)](https://fanurs.github.io/data-analysis-e15190-e14030/build/html/manualdoc/installation.html).
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
- [**`e15190/`**](e15190/): Core Python source codes for doing calibration and analysis
- [**`database/`**](database/): For storing all the calibration parameters, cache files (e.g. `.h5` files), images, ROOT files, Mako templates, etc.
- [**`local/`**](local/): Miscellaneous local configuration files, scripts, etc. C++ source codes also go into here, mainly to separate them from the Python source codes in [`e15190/`](e15190/).
- [**`scripts/`**](scripts/): Here are all the calibration scripts and batch scripts (for parallel computing).
- [**`tests/`**](tests): Contains all the test scripts.
- **`env_e15190/`**: Where the conda environment is built. This directory should not be committed to git. Any custom modifications should be added as symbolic links directing to [`local/`](local/).
- [**`docs/`**](docs/): Documentation of the project. This directory is used to build GitHub Pages for this project (see [here](https://fanurs.github.io/data-analysis-e15190-e14030/)), and the files are auto-generated using [Sphinx](https://www.sphinx-doc.org/).
- [**`environment.yml`**](environment.yml): Configuration file for setting up the conda environment.
- [**`build.py`**](build.py): Installation script. To build the conda environment as well as modifying a few other things, e.g. environment variables, terminal commands, etc.

## 3. Testing framework
We are using the [`pytest`](https://docs.pytest.org/) framework. To test everything, simply activate the conda environment, go to the project directory and type:
```console
pytest
```
To test a specify file or directory, e.g. `tests/utilities/test_timer.py`:
```console
pytest tests/utilities/test_timer.py
```

**When to do testing?**
* Right after installation, run the test to see if things are working correctly.
* After modifying any code, run the test to check in case things that were previously working are now broken. This is known as the "regression testing". [<b style="color: red;">Important!</b>]
* You are also encouraged to write and run tests *while* developing the source code. This is often considered as the best practice, though tedious.
* Any other scenarios where you think there might be a chance to break things, i.e. server updates, conda environment updates, git merge with other collaborators, etc.


## 4. PEP 8 style guide
The [PE P8 Style Guide](https://www.python.org/dev/peps/pep-0008/) is the most popular style guide among Python developers. It is a set of guidelines for writing Python code that aims to be consistent and readable. The webpage at https://www.python.org/dev/peps/pep-0008/ has a detailed description of the style guide. While it is always a good idea to go through the guidelines, not everyone will have the luxury to read the whole document and remember all the rules. So oftentimes, people use some automated tools to format their code.

In this project, we use [autopep8](https://pypi.org/project/autopep8/) to automatically format our code in order to comply with the PEP 8 standard. This Python package is already listed in [`environment.yml`](environment.yml), so right after you activate the conda environment, you can simply type:
```console
autopep8 demo_script.py --in-place
```
This command should have automatically formatted the `demo_script.py` script in place. In most cases, autopep8 only changes the style of the code, e.g. whitespaces, indentation, etc., and it should not change the behavior of the code. Always re-run some tests if you are not sure.

Lastly, it is strongly recommended to apply autopep8 to your script before committing to git or pushing to GitHub.
