#/bin/bash

# This script runs the 4 tested models (2 for mnist, 2 for cifar10) with recommended
# default hyperparameters and no pruning i.e. a regular training run to serve as 
# a baseline for comparing relative accuracy drop of pruned vs. unpruned.

# ResNet18
CUDA_VISIBLE_DEVICES=1 python main.py -m resnet18 -d cifar10 -bs 128 -e 350 -lr 0.1 --prune_criterion topflip \
                --prune_freq 14 --prune_rate 0.2 --opt sgd --momentum 0.9 --noise --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --stop_pruning_at 250 --use_ema_flips --beta_ema_flips 0.9 --milestones 150 250 \
                --logdir="baselines" --comment="resnet18 topflip original hparams +noise+ema stop@250"

# VGG19
CUDA_VISIBLE_DEVICES=1 python main.py -m vgg19 -d cifar10 -bs 128 -e 350 -lr 0.1 --prune_criterion topflip \
                --prune_freq 14 --prune_rate 0.2 --opt sgd --momentum 0.9 --noise --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --stop_pruning_at 250 --use_ema_flips --beta_ema_flips 0.9 --milestones 150 250 \
                --logdir="baselines" --comment="vgg19 topflip original hparams +noise+ema stop@250"
