CUDA_VISIBLE_DEVICES=1 python main.py -m resnet18 -d cifar10 -bs 128 -e 350 -lr 0.1 --prune_criterion global_magnitude \
                --prune_freq 2 --prune_rate 0.2 --opt sgd --momentum 0.9 --noise --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --use_ema_flips --beta_ema_flips 0.9 --milestones 150 250 \
                --logdir="test" --comment="vgg19 topflip ema_flips beta=0.9"