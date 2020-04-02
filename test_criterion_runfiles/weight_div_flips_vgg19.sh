#!/bin/bash
#SBATCH -t 16:00:00
#SBATCH -N 1
#SBATCH -p gpu_shared
#SBATCH --mem=16000M
#SBATCH --gres=gpu:1
#SBATCH --output=out_files/weight_div_flips_vgg19
source activate base
device=0;
seed=42;
echo "Running weight div flips test.";
for prune_freq in 50 39; do
    CUDA_VISIBLE_DEVICES=${device} python main.py -m vgg19 -d cifar10 -bs 128 -tbs 5000 -e 350 -lr 0.1  \
                --prune_criterion weight_div_flips --prune_rate 0.5 --prune_freq ${prune_freq} --seed ${seed} \
                --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 --noise --logdir="test" \
                --comment="vgg19 crit=weight_div_flips pf=${prune_freq} seed=${seed}"
done

seed=43;
echo "Running weight div flips test.";
for prune_freq in 50 39; do
    CUDA_VISIBLE_DEVICES=${device} python main.py -m vgg19 -d cifar10 -bs 128 -tbs 5000 -e 350 -lr 0.1  \
                --prune_criterion weight_div_flips --prune_rate 0.5 --prune_freq ${prune_freq} --seed ${seed} \
                --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 --noise --logdir="test" \
                --comment="vgg19 crit=weight_div_flips pf=${prune_freq} seed=${seed}"
done