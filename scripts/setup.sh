# Please run this script on bash

apt install python3-pip
pip3 install -U pip setuptools

pip3 install virtualenv

virtualenv venv

source venv/bin/activate

pip install .

# curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
# bash Anaconda3-2019.03-Linux-x86_64.sh
# conda create -n conda_venv python=3.7.3 anaconda
# conda activate conda_venv
# pip install setuptools
# python -m pip install .