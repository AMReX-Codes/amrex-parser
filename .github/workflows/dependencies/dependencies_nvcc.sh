#!/usr/bin/env bash

set -eu -o pipefail

# `man apt.conf`:
#   Number of retries to perform. If this is non-zero APT will retry
#   failed files the given number of times.
echo 'Acquire::Retries "3";' | sudo tee /etc/apt/apt.conf.d/80-retries

sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get -qqq update
sudo apt-get install -y \
    build-essential     \
    wget                \
    cmake               \
    g++

source /etc/os-release # set VERSION_ID
distro=ubuntu$(sed 's/\.//' <<< ${VERSION_ID})
wget https://developer.download.nvidia.com/compute/cuda/repos/${distro}/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
VERSION_DASHED=$(apt-cache search cuda-compiler* | tail -n1 | cut -d' '  -f1 | sed 's/cuda-compiler-//')
VERSON_DOTTED=$(sed 's/-/\./' <<< $VERSION_DASHED)
sudo apt-get install -y \
    cuda-command-line-tools-$VERSION_DASHED \
    cuda-compiler-$VERSION_DASHED           \
    cuda-minimal-build-$VERSION_DASHED
sudo ln -s cuda-$VERSION_DOTTED /usr/local/cuda
