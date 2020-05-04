#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_titanrtx_shared
#SBATCH --gres=gpu:1
#SBATCH --mem=16000M
#SBATCH --cpus-per-task=1
#SBATCH --job-name=ghs_resnet18
#SBATCH --output=out_files/ghs_resnet18.out
source activate base

python main.py --model resnet18 --dataset cifar10 -bs 128 -tbs 1000 -e 500 -lr 0.1 \
                --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 \
                --prune_criterion threshold --prune_freq 350 --magnitude_threshold 1e-4 \
                --add_ghs --hoyer_lambda 2e-4 --stop_hoyer_at 350 \
                --seed 42 \
                --comment=resnet18 \
                --logdir=ghs_test/resnet18