#!/bin/bash
#SBATCH -t 72:00:00
#SBATCH -N 1
#SBATCH -p gpu_shared
#SBATCH --mem=16000M
#SBATCH --gres=gpu:1
source activate base
device=0;
seed=42;
echo "Running $1"

for prune_freq in 50 39; do
# Do flips
    CUDA_VISIBLE_DEVICES=${device} python main.py -m resnet18 -d cifar10 -bs 128 -tbs 2000 -e 350 -lr 0.1  \
                --prune_criterion topflip --prune_rate 0.5 --prune_freq ${prune_freq} --seed ${seed} \
                --opt sgd --momentum 0.9 --noise --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 --logdir="scaling_noise_experiment" \
                --comment="resnet18 crit=topflip pf=${prune_freq} seed=${seed} noise_scale=10" \
                --noise_scale_factor 0.5

    CUDA_VISIBLE_DEVICES=${device} python main.py -m resnet18 -d cifar10 -bs 128 -tbs 2000 -e 350 -lr 0.1  \
                --prune_criterion topflip --prune_rate 0.5 --prune_freq ${prune_freq} --seed ${seed} \
                --opt sgd --momentum 0.9 --noise --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 --logdir="scaling_noise_experiment" \
                --comment="resnet18 crit=topflip pf=${prune_freq} seed=${seed} noise_scale=0.1" \
                --noise_scale_factor 0.25

done