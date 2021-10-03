#!/bin/bash

# reset all builds
make clean
rm -f source/autodoc/*
rm -rf source/_autosummary/*

# auto-generate *.rst files to autodoc
sphinx-apidoc ../e15190/ -o source/autodoc -f -e

# add "e15190" to index.rst after line ":recursive:"
sed -i -z 's/\:recursive\:/\:recursive\:\n      e15190/' source/index.rst

# generate html files for the first time
make html

# empty up build/ and source/autodoc/
make clean
rm -f source/autodoc/*

# remove the line "e15190" from index.rst
sed -i -z 's/\:recursive\:\n      e15190/\:recursive\:/' source/index.rst

# General the final html files using _autosummary/*.rst
make html
