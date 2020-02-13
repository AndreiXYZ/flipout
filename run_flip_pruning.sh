#/bin/bash

# This script runs the 4 tested models (2 for mnist, 2 for cifar10) with recommended
# default hyperparameters and flip pruning.

#For flips: rmsprop opt, wdecay, no momentum, noise

# LeNet300
CUDA_VISIBLE_DEVICES=1 python main.py -m lenet300 -d mnist -bs 256 -e 100 -lr 1e-3 --prune_criterion flip \
                --flip_threshold 25 --noise --opt rmsprop --reg_type wdecay --lambda 0\
                --anneal_lambda --logdir="baselines" --comment="lenet300 flips lambda=0(orig) thresh=25"
# LeNet5
CUDA_VISIBLE_DEVICES=1 python main.py -m lenet5 -d mnist -bs 256 -e 100 -lr 2e-3 --prune_criterion flip \
                --flip_threshold 25 --noise --opt rmsprop --reg_type wdecay --lambda 0\
                --anneal_lambda --logdir="baselines" --comment="lenet5 flips lambda=0(orig) thresh=25"

# ResNet18
CUDA_VISIBLE_DEVICES=1 python main.py -m resnet18 -d cifar10 -bs 128 -e 350 -lr 0.1 --prune_criterion flip \
                --flip_threshold 25 --noise --opt rmsprop --momentum 0 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --anneal_lambda --milestones 150 250 \
                --logdir="baselines" --comment="resnet18 lambda=5e-4(orig) thresh=25"

# VGG19
CUDA_VISIBLE_DEVICES=1 python main.py -m vgg19 -d cifar10 -bs 128 -e 350 -lr 0.1 --prune_criterion flip \
                --flip_threshold 25 --noise --opt rmsprop --momentum 0 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --anneal_lambda --milestones 150 250 \
                --logdir="baselines" --comment="vgg19 lambda=5e-4(orig) thresh=25"
