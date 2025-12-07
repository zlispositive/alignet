#!/usr/bin/env bash

# 1) Point to your private GCC
export GCC_HOME="$HOME/tools/gcc-13.2.0"
export PATH="$GCC_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$GCC_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 2) Tell build tools which compiler to use
export CC=gcc
export CXX=g++

# 3) Activate the Python virtualenv for this project
source .venv-alignet/bin/activate

# Optional debug prints
echo "Using GCC at: $(which gcc)"
gcc --version | head -n 1
echo "Python: $(which python)"
