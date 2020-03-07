# LeNet300
CUDA_VISIBLE_DEVICES=0 python main.py -m lenet300 -d mnist -bs 256 -e 200 -lr 1e-3 --prune_criterion topflip \
                --prune_freq 15 --prune_rate 0.3 --opt adam --noise --reg_type wdecay --lambda 3e-3\
                --logdir="mnist_experiments" --comment="lenet300 wdecay 3e-3"


# LeNet5
CUDA_VISIBLE_DEVICES=0 python main.py -m lenet5 -d mnist -bs 256 -e 200 -lr 2e-3 --prune_criterion topflip \
                --prune_freq 15 --prune_rate 0.3 --opt adam --noise --reg_type wdecay --lambda 3e-3\
                --logdir="mnist_experiments" --comment="lenet5 wdecay 3e-3"