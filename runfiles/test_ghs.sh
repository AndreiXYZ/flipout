#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_titanrtx_shared
#SBATCH --gres=gpu:1
#SBATCH --mem=16000M
#SBATCH --cpus-per-task=1
#SBATCH --job-name=test_structured_pruning
#SBATCH --output=out_files/test_structured_pruning.out
source activate base

python main.py --model vgg13 --dataset cifar10 -bs 128 -tbs 5000 -e 10 -lr 0.1 \
                --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 \
                --prune_criterion structured_magnitude --prune_freq 2 --prune_rate 0.1 \
                --prune_bias --prune_bnorm \
                --seed 42 \
                --comment=resnet18 \
                --logdir=ghs_test/resnet18
