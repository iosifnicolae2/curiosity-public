#!/bin/bash

source ./venv/bin/activate

cd ml-agents || exit
pip3 install -e ./

cd ..
pip install -r requirements.txt


