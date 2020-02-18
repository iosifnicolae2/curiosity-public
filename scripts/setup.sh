# Please run this script on bash

apt install python3-pip
pip3 install -U pip setuptools

pip3 install virtualenv

virtualenv venv

source venv/bin/activate

pip install .

# conda create -n conda_venv python=3.7.3 anaconda
# conda activate conda_venv
# pip install setuptools
# python -m pip install .