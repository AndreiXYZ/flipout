#!/bin/bash
device=1
seed=42

for prune_freq in 2; do
    # Do magnitude and random
    # for prune_criterion in 'global_magnitude';do
    #     # ResNet18
    #     CUDA_VISIBLE_DEVICES=${device} python main.py -m resnet18 -d cifar10 -bs 128 -tbs 10000 -e 350 -lr 0.1 \
    #                     --prune_criterion ${prune_criterion}  --prune_rate 0.2 --prune_freq ${prune_freq} --seed ${seed} \
    #                     --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
    #                     --milestones 150 250 --logdir="test" \
    #                     --comment="test_rn18_myrun"
    # done
        CUDA_VISIBLE_DEVICES=1 python main.py -m resnet18 -d cifar10 -bs 128 -e 10 -lr 0.1 \
                --prune_criterion global_magnitude --prune_freq 2 --prune_rate 0.2 --seed ${seed} \
                --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 --logdir="test" --comment="vgg19 topflip ema_flips beta=0.9"
    
done
