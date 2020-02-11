#/bin/bash

# This script runs the 4 tested models (2 for mnist, 2 for cifar10) with recommended
# default hyperparameters and flip pruning.

#For flips: rmsprop opt, wdecay, no momentum, add noise

# LeNet300
python main.py -m lenet300 -d mnist -bs 256 -e 100 -lr 1e-3 --prune_criterion flip \
                --flip_threshold 25 --opt rmsprop --noise --reg_type wdecay --lambda 0\
                --anneal_lambda --logdir="baselines" --comment="lenet300 flips lambda=0(orig) thresh=25" \

# LeNet5
python main.py -m lenet5 -d mnist -bs 256 -e 100 -lr 2e-3 --prune_criterion flip \
                --flip_threshold 25 --opt rmsprop --noise --reg_type wdecay --lambda 0\
                --anneal_lambda --logdir="baselines" --comment="lenet5 flips lambda=0(orig) thresh=25" \

# ResNet18
python main.py -m resnet18 -d cifar10 -bs 128 -e 350 -lr 0.1 --prune_criterion flip \
                --flip_threshold 25 --opt rmsprop --momentum 0 --reg_type wdecay --lambda 0 --use_scheduler \
                --anneal_lambda --noise --milestones 150 250 \
                --logdir="baselines" --comment="resnet18 lambda=0 thresh=25" \

# VGG19
python main.py -m vgg19 -d cifar10 -bs 128 -e 350 -lr 0.1 --prune_criterion flip \
                --flip_threshold 25 --opt rmsprop --momentum 0 --reg_type wdecay --lambda 0 --use_scheduler \
                ---anneal_lambda -noise --milestones 150 250 \
                --logdir="baselines" --comment="vgg19 lambda=0 thresh=25" \