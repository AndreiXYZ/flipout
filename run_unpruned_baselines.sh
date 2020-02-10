#/bin/bash

# This script runs the 4 tested models (2 for mnist, 2 for cifar10) with recommended
# default hyperparameters and no pruning i.e. a regular training run to serve as 
# a baseline for comparing relative accuracy drop of pruned vs. unpruned.

# LeNet300
python main.py -m lenet300 -d mnist -bs 256 -e 100 -lr 1e-3 --prune_criterion none \
                --opt adam --logdir="baselines" --comment="lenet300 unrpuned"

# LeNet5
python main.py -m lenet5 -d mnist -bs 256 -e 100 -lr 2e-3 --prune_criterion none \
                --opt adam --logdir="baselines" --comment="lenet5 unpruned"

# ResNet18
python main.py -m resnet18 -d cifar10 -bs 128 -e 350 -lr 0.1 --prune_criterion none \
                --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 --logdir="baselines" --comment="resnet18 unpruned"

# VGG19
python main.py -m vgg19 -d cifar10 -bs 128 -e 350 -lr 0.1 --prune_criterion none \
                --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                 --milestones 150 250 --logdir="baselines" --comment="vgg19 unpruned"

