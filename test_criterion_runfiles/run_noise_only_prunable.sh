#!/bin/bash
device=1;
seed=42;

for prune_freq in 117 70 50 39; do
    #Do flips while adding noise only to prunable params
    CUDA_VISIBLE_DEVICES=${device} python main.py -m resnet18 -d cifar10 -bs 128 -tbs 10000 -e 350 -lr 0.1  \
                --prune_criterion topflip --prune_rate 0.5 --prune_freq ${prune_freq} --seed ${seed} \
                --opt sgd --momentum 0.9 --noise --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 --logdir="topflip_ablations" \
                --comment="resnet18 crit=topflip pf=${prune_freq} seed=${seed}" \
                --noise_only_prunable \
                --save_model "pre-finetune/resnet18_topflip_pf${prune_freq}_s${seed}_noise_only_pruable"

    # Use EMA flips
    CUDA_VISIBLE_DEVICES=${device} python main.py -m resnet18 -d cifar10 -bs 128 -tbs 10000 -e 350 -lr 0.1  \
                --prune_criterion topflip --prune_rate 0.5 --prune_freq ${prune_freq} --seed ${seed} \
                --opt sgd --momentum 0.9 --noise --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 --logdir="topflip_ablations" \
                --comment="resnet18 crit=topflip pf=${prune_freq} seed=${seed}" \
                --use_ema_flips --beta_ema 0.9 \
                --save_model "pre-finetune/resnet18_topflip_pf${prune_freq}_s${seed}_noise_only_pruable"

    # Use both
    CUDA_VISIBLE_DEVICES=${device} python main.py -m resnet18 -d cifar10 -bs 128 -tbs 10000 -e 350 -lr 0.1  \
                --prune_criterion topflip --prune_rate 0.5 --prune_freq ${prune_freq} --seed ${seed} \
                --opt sgd --momentum 0.9 --noise --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 --logdir="topflip_ablations" \
                --comment="resnet18 crit=topflip pf=${prune_freq} seed=${seed}" \
                --use_ema_flips --beta_ema 0.9 --noise_only_prunable\
                --save_model "pre-finetune/resnet18_topflip_pf${prune_freq}_s${seed}_noise_only_pruable"

done