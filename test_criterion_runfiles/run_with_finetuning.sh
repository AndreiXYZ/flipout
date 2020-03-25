#!/bin/bash
#SBATCH -t 72:00:00
#SBATCH -N 1
#SBATCH -p gpu_shared
#SBATCH --mem=16000M
#SBATCH --gres=gpu:1

source activate base
device=0;
seed=42;
prune_freq=50;
#Flips
CUDA_VISIBLE_DEVICES=${device} python main.py -m resnet18 -d cifar10 -bs 128 -tbs 2000 -e 100 -lr 0.001  \
            --prune_criterion none --seed ${seed} \
            --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 \
            --logdir="test" \
            --comment="resnet18 crit=magnitude pf=50 seed=${seed} finetuned" \
            --load_model "/home/andreia/thesis/chkpts/criterion_experiment_no_bias/pre-finetune/resnet18_topflip_pf50_s42.pt" \
            --save_model "criterion_experiment_no_bias/post-finetune/resnet18_topflip_pf${prune_freq}_s${seed}_noise_only_prunable" \
# Magnitude                
CUDA_VISIBLE_DEVICES=${device} python main.py -m resnet18 -d cifar10 -bs 128 -tbs 2000 -e 100 -lr 0.001  \
            --prune_criterion none --seed ${seed} \
            --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 \
            --logdir="test" \
            --comment="resnet18 crit=topflip pf=${prune_freq} seed=${seed} finetuned" \
            --load_model "/home/andreia/thesis/chkpts/criterion_experiment_no_bias/pre-finetune/resnet18_global_magnitude_pf50_s42.pt" \
            --save_model "criterion_experiment_no_bias/post-finetune/resnet18_topflip_pf${prune_freq}_s${seed}_noise_only_prunable" \


