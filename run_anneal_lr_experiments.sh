#/bin/bash

# This script runs the 4 tested models (2 for mnist, 2 for cifar10) with hparams
# as close as possible to default.

#For this experiment: no noise, threshold 1, anneal lr, large lr, no wdecay anneal

# LeNet300
CUDA_VISIBLE_DEVICES=1 python main.py -m lenet300 -d mnist -bs 256 -e 100 -lr 1e-1 --prune_criterion flip \
                --flip_threshold 1 --opt rmsprop --reg_type wdecay --lambda 0\
                --anneal_lr --logdir="baselines" --comment="lenet300 flips lambda=0(orig) thresh=1" 
# LeNet5
CUDA_VISIBLE_DEVICES=1 python main.py -m lenet5 -d mnist -bs 256 -e 100 -lr 2e-1 --prune_criterion flip \
                --flip_threshold 1 --opt rmsprop --reg_type wdecay --lambda 0\
                --anneal_lr --logdir="baselines" --comment="lenet5 flips lambda=0(orig) thresh=1" 

# ResNet18
CUDA_VISIBLE_DEVICES=1 python main.py -m resnet18 -d cifar10 -bs 128 -e 350 -lr 0.1 --prune_criterion flip \
                --flip_threshold 1 --opt rmsprop --momentum 0 --reg_type wdecay --lambda 0 \
                --anneal_lr --logdir="baselines" --comment="resnet18 lambda=0 thresh=1" 

# VGG19
CUDA_VISIBLE_DEVICES=1 python main.py -m vgg19 -d cifar10 -bs 128 -e 350 -lr 0.1 --prune_criterion flip \
                --flip_threshold 1 --opt rmsprop --momentum 0 --reg_type wdecay --lambda 0 \
                --anneal_lr --logdir="baselines" --comment="vgg19 lambda=0 thresh=1" 