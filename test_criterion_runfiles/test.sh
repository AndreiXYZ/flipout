#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -p gpu_titanrtx_shared
#SBATCH --mem=16000M
#SBATCH --gres=gpu:1
#SBATCH --output=out_files/densenet_imagenette_test.txt
source activate base
device=0;
prune_criterion='magnitude';

seed=42;
# Test densenet121
CUDA_VISIBLE_DEVICES=${device} python main.py -m densenet121 -d imagenette -bs 128 -tbs 512 -e 350 -lr 0.1 \
                --seed ${seed} --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 --logdir="test" \
                --comment="densenet161 test"