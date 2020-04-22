#!/bin/bash
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH -p gpu_titanrtx_shared
#SBATCH --mem=16000M
#SBATCH --output misc/vgg19_stats.txt

source activate base
#not needed: SBATCH --gres=gpu:1

python misc/read_tb_logs.py --log_folder runs/criterion_experiment_no_bias/vgg19