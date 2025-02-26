#!/bin/bash

# Update package lists
# sudo apt update

# Install system dependencies
sudo apt install -y \
    git \
    python3 \
    python3-venv \
    python3-pip \
    build-essential \
    libsuitesparse-dev \
    libopenblas-dev \
    swig \
    pkg-config \
    cmake \
    tmux

# Clone the scikit-umfpack repository
git clone https://github.com/scikit-umfpack/scikit-umfpack.git
cd scikit-umfpack

# Create a virtual environment named 'sc'
python3 -m venv sc
source sc/bin/activate

# Upgrade pip
# pip install --upgrade pip

# Install specific version of numpy and the latest scipy
pip install numpy==1.26.4 scipy matplotlib meshio --index-url = http://mirrors.aliyun.com/pypi/simple/

# Install scikit-umfpack from the cloned repository
pip install .

# Deactivate the virtual environment
deactivate

# Navigate out of the repository directory
cd ..

echo "scikit-umfpack has been successfully installed in the 'sc' virtual environment."
