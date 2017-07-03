#!/bin/bash
echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
if ! dpkg-query -W cuda; then
  # The 16.04 installer works with 16.10.
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  apt-get update
  apt-get install cuda -y
fi


export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export CUDA_HOME=/usr/local/cuda-8.0
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64\
                     ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
                     