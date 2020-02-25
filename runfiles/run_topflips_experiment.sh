#/bin/bash

# LeNet300
# CUDA_VISIBLE_DEVICES=0 python main.py -m lenet300 -d mnist -bs 256 -e 200 -lr 1e-3 --prune_criterion topflip \
#                 --use_ema_flips --beta_ema_flips 0.5 --prune_freq 8 --prune_rate 0.2 --opt adam --noise \
#                 --reg_type wdecay --lambda 3e-3 \
#                 --logdir="baselines" --comment="lenet300 topflip original +noise+ema+wdecay beta=0.5"

# # LeNet5
# CUDA_VISIBLE_DEVICES=0 python main.py -m lenet5 -d mnist -bs 256 -e 200 -lr 2e-3 --prune_criterion topflip \
#                 --use_ema_flips --beta_ema_flips 0.5 --prune_freq 8 --prune_rate 0.2 --opt adam --noise \
#                 --reg_type wdecay --lambda 3e-3 \
#                 --logdir="baselines" --comment="lenet5 topflip original hparams +noise+ema+wdecay beta=0.5"

# ResNet18
CUDA_VISIBLE_DEVICES=0 python main.py -m resnet18 -d cifar10 -bs 128 -e 350 -lr 0.1 --prune_criterion topflip \
                --prune_freq 14 --prune_rate 0.2 --opt sgd --momentum 0.9 --noise --scale_noise_by_lr \
                --reg_type wdecay --lambda 5e-4 --use_scheduler --use_ema_flips --beta_ema_flips 0.9 --milestones 150 250 \
                --logdir="baselines" --comment="resnet18 topflip original hparams +scaled_noise+ema beta=0.9"

# VGG19
CUDA_VISIBLE_DEVICES=0 python main.py -m vgg19 -d cifar10 -bs 128 -e 350 -lr 0.1 --prune_criterion topflip \
                --prune_freq 14 --prune_rate 0.2 --opt sgd --momentum 0.9 --noise --scale_noise_by_lr \
                --reg_type wdecay --lambda 5e-4 --use_scheduler --use_ema_flips --beta_ema_flips 0.9 --milestones 150 500 \
                --logdir="baselines" --comment="vgg19 topflip original hparams +scaled_noise+ema beta=0.9"
