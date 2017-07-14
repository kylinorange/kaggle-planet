#!/bin/bash

sudo pip3 install -U keras

# sudo mkdir /data
# sudo mount /dev/xvdf /data

jupyter notebook --generate-config

cd /data/kaggle-planet
jupyter notebook --no-browser --port=8888 --ip=*
