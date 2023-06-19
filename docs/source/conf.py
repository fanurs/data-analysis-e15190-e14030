# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from e15190 import PROJECT_DIR
sys.path.insert(0, os.path.abspath(str(PROJECT_DIR / 'e15190')))

# -- Project information -----------------------------------------------------

project = 'Data Analysis for E15190-E14030'
copyright = '2018-2023, MSU/FRIB HiRA group'
author = 'Fanurs, et al.'

# The full version, including alpha/beta/rc tags
# release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    # 'sphinx.ext.viewcode',
]
autosummary_generate = True


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinxdoc'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Customization -----------------------------------------------------------

# To include docstrings from public methods and special methods
# autoclass_content = 'both'
autodoc_default_options = {
    'members': True,
    # 'undoc-members': True,
    'private-members': True,
}

# To preserve the order of the methods
autodoc_member_order = 'bysource'

# To disable "Show source" links to the *.rst files
html_show_sourcelink = False
