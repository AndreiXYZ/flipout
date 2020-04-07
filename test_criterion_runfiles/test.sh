#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -p gpu_titanrtx_shared
#SBATCH --mem=16000M
#SBATCH --gres=gpu:1
#SBATCH --output=out_files/weight_sq_div_flips_test.txt
source activate base
device=0;
prune_criterion='magnitude';

seed=42;
# Test weights squared divided by flips
CUDA_VISIBLE_DEVICES=${device} python main.py -m vgg19 -d cifar10 -bs 128 -tbs 5000 -e 350 -lr 0.1 \
                --prune_criterion weight_squared_div_flips --prune_rate 0.5 --prune_freq 2 \
                --seed ${seed} --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 \
                --noise --use_scheduler --milestones 150 250 \
                --logdir="test" \
                --comment="weight squared div flips test"