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

source /etc/os-release # set UBUNTU_CODENAME
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID}/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y \
    cuda-command-line-tools \
    cuda-compiler           \
    cuda-minimal-build
