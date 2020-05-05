#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_titanrtx_shared
#SBATCH --gres=gpu:1
#SBATCH --mem=16000M
#SBATCH --cpus-per-task=1
#SBATCH --job-name=structured_magnitude
#SBATCH --output=out_files/structured_magnitude.out
source activate base

python main.py --model vgg19 --dataset cifar10 -bs 128 -tbs 5000 -e 100 -lr 0.1 \
                --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 \
                --prune_criterion structured_magnitude \
                --prune_freq 30 --prune_rate 0.2 --prune_bias \
                --seed 42 \
                --comment=test \
                --logdir=structured_magnitude/resnet18