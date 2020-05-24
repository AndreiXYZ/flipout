#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --partition=gpu_titanrtx_shared
#SBATCH --gres=gpu:1
#SBATCH --mem=16000M
#SBATCH --cpus-per-task=1
#SBATCH --job-name=wsqdivflips_prune_like_hoyer
#SBATCH --output=out_files/wsqdivflips_prune_like_hoyer_44.out
source activate base

python main.py --model densenet121 --dataset imagenette -bs 128 -tbs 1000 -e 500 -lr 0.1 \
                --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 \
                --prune_criterion weight_squared_div_flips --prune_freq 350 --prune_rate 0.9990 \
                --seed 44 --noise \
                --comment=flipout_prune_like_hoyer \
                --logdir=test