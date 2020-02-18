# Please run this script on bash

apt install python3-pip
apt-get install python3-setuptools
apt-get install python-setuptools
pip install -U pip setuptools

pip3 install virtualenv

virtualenv venv

source venv/bin/activate

pip install .