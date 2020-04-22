#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --partition=gpu_titanrtx_shared
#SBATCH --gres=gpu:1
#SBATCH --mem=16000M
#SBATCH --cpus-per-task=1
#SBATCH --job-name=global_magnitude_noise
#SBATCH --output=out_files/global_magnitude_noise.out
source activate base

python main.py --model resnet18 --dataset cifar10 -bs 128 -tbs 5000 -e 350 -lr 0.1 \
                --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 \
                --prune_criterion global_magnitude --prune_freq 32 --prune_rate 0.5 \
                --seed 42 --noise \
                --comment=resnet18_global_mag_noise \
                --logdir=test