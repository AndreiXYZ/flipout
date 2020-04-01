#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -p gpu_shared
#SBATCH --mem=16000M
#SBATCH --gres=gpu:1

source activate base
device=0;
prune_criterion='magnitude';

seed=43;
for prune_freq in 117 70 50 39; do
    # Do magnitude
    CUDA_VISIBLE_DEVICES=${device} python main.py -m resnet18 -d cifar10 -bs 128 -tbs 2000 -e 350 -lr 0.1 \
                    --prune_criterion ${prune_criterion}  --prune_rate 0.5 --prune_freq  ${prune_freq} --seed ${seed} \
                    --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                    --milestones 150 250 --logdir="criterion_experiment_no_bias/resnet18" \
                    --comment="resnet18 crit=${prune_criterion} pf=${prune_freq} seed=${seed}" \
                    --save_model "pre-finetune/resnet18_${prune_criterion}_pf${prune_freq}_s${seed}"

done

seed=44;
for prune_freq in 117 70 50 39; do
    # Do magnitude
    CUDA_VISIBLE_DEVICES=${device} python main.py -m resnet18 -d cifar10 -bs 128 -tbs 2000 -e 350 -lr 0.1 \
                    --prune_criterion ${prune_criterion}  --prune_rate 0.5 --prune_freq  ${prune_freq} --seed ${seed} \
                    --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                    --milestones 150 250 --logdir="criterion_experiment_no_bias/resnet18" \
                    --comment="resnet18 crit=${prune_criterion} pf=${prune_freq} seed=${seed}" \
                    --save_model "pre-finetune/resnet18_${prune_criterion}_pf${prune_freq}_s${seed}"

done