.. _homepage:
.. highlight:: console
.. toctree::
   :maxdepth: 2
   :caption: Contents:
.. autosummary::
   :toctree: _autosummary
   :template: custom-module.rst
   :recursive:
      e15190
      e15190

**********************************************
Documentation for Data Analysis: E15190-E14030
**********************************************

Useful links
============

- `GitHub repository <https://github.com/Fanurs/data-analysis-e15190-e14030>`__
- `Plotly: Infographics <https://groups.nscl.msu.edu/hira/15190-14030/fanurs/index.html>`__
- `Run log (WMU) <http://neutronstar.physics.wmich.edu/runlog/index.php?op=list>`__
- `Experiment homepage (original) <https://groups.nscl.msu.edu/hira/15190-14030/index.htm>`__
- :doc:`API reference <./_autosummary/e15190>`
- :ref:`modindex`

Documentation
=============

All the Python source code is documented at :doc:`here <./_autosummary/e15190>`.

C++ source code and standalone scripts are currently not being documented.

Background
==========

This repository primarily focuses on neutron wall analysis. Instead of starting from the raw event files (binary files from the DAQ), we treat the ROOT files generated using `Daniele's analysis framework <https://github.com/nscl-hira/E15190-Unified-Analysis-Framework>`__ (currently using branch ``zhu``) as inputs, and from there we write our own analysis codes. In the case where calibration parameters are not good enough or some minor mistakes were found in the framework, we always try *not* to modify the framework. Instead, we would like to "freeze" the framework, and correct for any imperfection or errors in the output ROOT files within this repository. Unless there is something that framework does not preserve in the ROOT files, e.g. missing events, then only we will go back and try to debug the framework (currently WMU is working on this).

This decision was made after considering the fact that all major authors of the framework had left the group. It is simply safer and more efficient to keep it as "legacy code", and build on top of the ROOT files it generated. Fortunately, most of the calibrations have already been done. While some are found to be off later on, most are already good enough for a simple "first-order" analysis.


Installation
===============

If you are not familiar with conda, more detailed instructions can be found at :ref:`Installation (detailed version) <installation-detailed-version>`.

1. Git clone the repository.
   ::
      git clone https://github.com/Fanurs/data-analysis-e15190-e14030.git

2. Install (if haven't) and activate ``conda``.

3. Build the conda environment. This step is *time-consuming* (~hours). Use a screen session whenever possible.
   ::
      cd data-analysis-e15190-e14030/
      ./build.py

4. Activate the conda environment.
   ::
      conda activate ./env_e15190/

5. To quickly check if the conda environment has been activated, inspect the environment variable ``$PROJECT_DIR``.
   ::
      echo $PROJECT_DIR


Structure of the repository
===========================

This repository is written mainly in Python 3.8 and C++20 (``-std=c++2a`` in GCC 9 and earlier).

- `e15190/ <https://github.com/Fanurs/data-analysis-e15190-e14030/tree/main/e15190>`__: Core Python source codes for doing calibration and analysis

- `database/ <https://github.com/Fanurs/data-analysis-e15190-e14030/tree/main/database>`__: For storing all the calibration parameters, cache files (e.g. `.h5` files), images, ROOT files, Mako templates, etc.

- `local/ <https://github.com/Fanurs/data-analysis-e15190-e14030/tree/main/local>`__: Miscellaneous local configuration files, scripts, etc. C++ source codes also go into here, mainly to separate them from the Python source codes in `e15190/ <https://github.com/Fanurs/data-analysis-e15190-e14030/tree/main/e15190>`__.

- `scripts/ <https://github.com/Fanurs/data-analysis-e15190-e14030/tree/main/scripts>`__: Here are all the calibration scripts and batch scripts (for parallel computing).

- `tests/ <https://github.com/Fanurs/data-analysis-e15190-e14030/tree/main/tests>`__: Contains all the test scripts.

- env_e15190/: Where the conda environment is built. This directory should not be committed to git. Any custom modifications should be added as symbolic links directing to `local/ <https://github.com/Fanurs/data-analysis-e15190-e14030/tree/main/local>`__.

- `docs/ <https://github.com/Fanurs/data-analysis-e15190-e14030/tree/main/docs>`__: Documentation of the project. This directory is used to build GitHub Pages for this project (see `here <https://fanurs.github.io/data-analysis-e15190-e14030/>`__, and the files are auto-generated using `Sphinx <https://www.sphinx-doc.org/>`__.

- `environment.yml <https://github.com/Fanurs/data-analysis-e15190-e14030/tree/main/environment.yml>`__: Configuration file for setting up the conda environment.

- `build.py <https://github.com/Fanurs/data-analysis-e15190-e14030/tree/main/build.py>`__: Installation script. To build the conda environment as well as modifying a few other things, e.g. environment variables, terminal commands, etc.


Testing framework
=================

We are using the `pytest <https://docs.pytest.org/>`__ framework. To test everything, simply activate the conda environment, go to the project directory and type:
::
   pytest

To test a specify file or directory, e.g. ``tests/utilities/test_timer.py``:
::
   pytest tests/utilities/test_timer.py

When to do unit testing?
------------------------
.. raw:: html
   <style> .red {color:red} </style>
.. role:: red
* Right after installation, run all tests to see if things are working correctly.
* After changing any codes, run all tests to check in case things that were previously working are now broken, a.k.a. regression testing. :red:`Important!`
* You are also encouraged to write and run tests *while* developing the source code. This is often considered as the best practice, though sometimes a little too tedious.
* Any other scenarios where you think there might be a chance to break things, i.e. server updates, conda environment updates, git merge with other collaborators, etc.


PEP 8 style guide
=================

The `PEP 8 Style Guide <https://www.python.org/dev/peps/pep-0008/>`__ is the most popular style guide among Python developers. It is a set of guidelines for writing Python code that aims to be consistent and readable. The `webpage <https://www.python.org/dev/peps/pep-0008/>`__ has a detailed description of the style guide. While it is always a good idea to go through the guidelines, not everyone will have the luxury to read the whole document and remember all the rules. So oftentimes, people use some automated tools to format their code.

In this project, we use `autopep8 <https://pypi.org/project/autopep8/>`__ to automatically format our code in order to comply with the PEP 8 standard. This Python package is already listed in `environment.yml <https://github.com/Fanurs/data-analysis-e15190-e14030/tree/main/environment.yml>`__, so right after you activate the conda environment, you can simply type:
::
   autopep8 demo_script.py --in-place

This command should have automatically formatted the ``demo_script.py`` script in place. In most cases, autopep8 only changes the style of the code, e.g. whitespaces, indentation, etc., and it should not change the behavior of the code. Always re-run some tests if you are not sure.

Lastly, it is strongly recommended to apply autopep8 to your script before committing to git or pushing to GitHub.


Further readings
================

- [Value-Assigned Pulse Shape Discrimination for Neutron Detectors](https://doi.org/10.1109/TNS.2021.3091126) by F. C. E. Teh, et al.
- [Calibration of large neutron detection arrays using cosmic rays](https://doi.org/10.1016/j.nima.2020.163826) by K. Zhu, et al.
- [Non-linearity effects on the light-output calibration of light charged particles in CsI(Tl) scintillator crystals](https://doi.org/10.1016/j.nima.2019.03.065) by D. Dell'Aquila, et al.
- [Reaction losses of charged particles in CsI(Tl) crystals](https://doi.org/10.1016/j.nima.2021.165798) by S. Sweany, et al.
- [Doctoral dissertation](https://groups.nscl.msu.edu/hira/Publications%20and%20Theses/Thesis_Sean_Sweany.pdf) by Sean Robert Sweany.
- [Doctoral dissertation](https://groups.nscl.msu.edu/hira/Publications%20and%20Theses/Zhu_dissertation.pdf) by Kuan Zhu.
- [Doctoral dissertation](https://publications.nscl.msu.edu/thesis/Coupland_2013_338.pdf) by Daniel David Schechtman Coupland
