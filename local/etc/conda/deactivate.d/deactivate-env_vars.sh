#!/bin/bash
if [ ! -z "$OLD_PROJECT_DIR" ]; then
    PROJECT_DIR=$OLD_PROJECT_DIR
    unset OLD_PROJECT_DIR
else
    unset PROJECT_DIR
fi
