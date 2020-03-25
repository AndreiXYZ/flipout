#!/bin/bash
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -p gpu_shared
#SBATCH --mem=16000M
#SBATCH --gres=gpu:1
source activate base

prune_freq=2;
device=0;
seed=42;

CUDA_VISIBLE_DEVICES=${device} python main.py -m resnet18 -d cifar10 -bs 128 -tbs 5000 -e 6 -lr 0.1 \
            --prune_criterion topflip --prune_rate 0.5 --prune_freq ${prune_freq} --seed ${seed} \
            --opt sgd --momentum 0.9 --noise --reg_type wdecay --lambda 5e-4 --use_scheduler \
            --milestones 150 250 --logdir="test" \
            --comment="resnet18 noise only prunable" \
            --noise_only_prunable \
            --save_model "pre-finetune/rn18_noise_only_prunable"

CUDA_VISIBLE_DEVICES=${device} python main.py -m resnet18 -d cifar10 -bs 128 -tbs 5000 -e 6 -lr 0.1 \
            --prune_criterion topflip --prune_rate 0.5 --prune_freq ${prune_freq} --seed ${seed} \
            --opt sgd --momentum 0.9 --noise --reg_type wdecay --lambda 5e-4 --use_scheduler \
            --milestones 150 250 --logdir="test" \
            --comment="resnet18 noise all" \
            --save_model "pre-finetune/rn18_noise_only_prunable"
