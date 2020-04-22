#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -p gpu_titanrtx_shared
#SBATCH --mem=16000M
#SBATCH --gres=gpu:1
#SBATCH --output=densenet_hs_lambda_1e-3_rerun_test.txt
source activate base
device=0;

seed=43;
# Test weights squared divided by flips
CUDA_VISIBLE_DEVICES=${device} python main.py -m densenet121 -d imagenette -bs 128 -tbs 1000 -e 350 -lr 0.1 \
                --prune_criterion global_magnitude --prune_rate 0.5 --prune_freq 32 \
                --seed ${seed} --opt sgd --momentum 0.9 --reg_type hs --lambda 0.001 \
                --use_scheduler --milestones 150 250 \
                --seed 42 --logdir="test" \
                --comment="rerun densenet experiment lambda=0.001 sp=99.9%"