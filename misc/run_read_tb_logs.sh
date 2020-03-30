#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -p gpu_shared
#SBATCH --mem=4000M
#SBATCH --gres=gpu:1
#SBATCH --output misc/output.txt

source activate base

python misc/read_tb_logs.py --log_folder runs/criterion_experiment_no_bias/vgg19