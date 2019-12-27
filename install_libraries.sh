#!/bin/bash

source ./venv/bin/activate

cd ml-agents || exit
pip3 install -e ./

cd ..
pip install -r requirements.txt


brew install graphviz
pip install graphviz
pip install tfgraphviz

cd gym-project || exit
brew install cmake openmpi

cd simple_gym_env || exit
pip install -e .

cd torch-ac || exit
pip install -e .
