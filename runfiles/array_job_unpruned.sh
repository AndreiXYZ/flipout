#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --partition=gpu_titanrtx_shared
#SBATCH --gres=gpu:1
#SBATCH --mem=16000M
#SBATCH --cpus-per-task=1
#SBATCH --job-name=vgg19_unpruned_44
#SBATCH --output=out_files/pretrained/vgg19unpruned_44.txt
source activate base
device=0;

CUDA_VISIBLE_DEVICES=${device} python main.py --model vgg19 --dataset cifar10 \
                -bs 128 -tbs 5000 -e 350 -lr 0.1 --prune_criterion none \
                --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 --seed 44 \
                --logdir=pretrained/vgg19 \
                --comment=vgg19_pretrained_seed=44 \
                --save_model=vgg19_pretrained_seed=44
