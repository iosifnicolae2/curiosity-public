cd app/sm_2d

python -m cProfile -o train.prof train.py

snakeviz train.prof