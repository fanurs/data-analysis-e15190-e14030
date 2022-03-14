#!/bin/bash

export PROJECT_DIR=`cd ../; pwd`

# When using Singularity, ROOT's Docker image
if [ -n $SINGULARITY_NAME ]; then
    echo "Detected Singularity container"
    return 0
fi

# When using local ROOT installations
if command -v module; then
    # e.g. when on a regular non-VS Code terminal
    module load root/gnu/6.24.02
else
    # typically happens when running from VS Code terminal
    echo "The command \"module load\" is not available."
    echo "Using hard-coded paths instead."

    source /mnt/misc/sw/x86_64/Debian/10/root/gnu/6.24.02/bin/thisroot.sh

    export PATH=/mnt/misc/sw/x86_64/all/gnu/gcc/9.3.0/bin/:$PATH
    export LD_LIBRARY_PATH=/mnt/misc/sw/x86_64/all/gnu/gcc/9.3.0/lib64/:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/mnt/misc/sw/x86_64/all/gnu/gcc/9.3.0/lib/:$LD_LIBRARY_PATH
    export CPLUS_INCLUDE_PATH=/mnt/misc/sw/x86_64/all/gnu/gcc/9.3.0/include:/mnt/misc/sw/x86_64/all/gnu/gcc/9.3.0/include/c++/9.3.0:/mnt/misc/sw/x86_64/all/gnu/gcc/9.3.0/include/c++/9.3.0/x86_64-linux-gnu:/mnt/misc/sw/x86_64/all/gnu/gcc/9.3.0/include/c++/9.3.0/backward
    export C_INCLUDE_PATH=/mnt/misc/sw/x86_64/all/gnu/gcc/9.3.0/include:/mnt/misc/sw/x86_64/all/gnu/gcc/9.3.0/include/c++/9.3.0:/mnt/misc/sw/x86_64/all/gnu/gcc/9.3.0/include/c++/9.3.0/x86_64-linux-gnu:/mnt/misc/sw/x86_64/all/gnu/gcc/9.3.0/include/c++/9.3.0/backward
fi