#/bin/bash

# ResNet18
CUDA_VISIBLE_DEVICES=0 python main.py -m resnet18 -d cifar10 -bs 128 -e 350 -lr 0.1 --prune_criterion topflip \
                --prune_freq 40 --prune_rate 0.2 --opt sgd --momentum 0.9 --noise --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --use_ema_flips --beta_ema_flips 0.9 --milestones 150 250 --logdir="baselines" --comment="resnet18 topflip pf=40 pr=0.2"

# VGG19
CUDA_VISIBLE_DEVICES=0 python main.py -m vgg19 -d cifar10 -bs 128 -e 350 -lr 0.1 --prune_criterion topflip \
                --prune_freq 40 --prune_rate 0.2 --opt sgd --momentum 0.9 --noise --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --use_ema_flips --beta_ema_flips 0.9 --milestones 150 250 --logdir="baselines" --comment="vgg19 topflip pf=40 pr=0.2"

