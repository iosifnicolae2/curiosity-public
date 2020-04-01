cd app/sm_2d

python -m cProfile -o train.prof train.py

snakeviz train.prof

# GPU proffiling
nvprof -o gpu.prof python train.py
snakeviz gpu.prof