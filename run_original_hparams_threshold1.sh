
# Also run VGG19 again cuz I had to do it. This run can be safely removed after
# VGG19
CUDA_VISIBLE_DEVICES=1 python main.py -m vgg19 -d cifar10 -bs 128 -e 350 -lr 0.001 --prune_criterion flip \
                --flip_threshold 25 --noise --opt rmsprop --momentum 0 --reg_type wdecay --lambda 5e-4 \
                --anneal_lambda \
                --logdir="baselines" --comment="vgg19 lambda=5e-4(orig) thresh=25 no scheduler"

# LeNet300
CUDA_VISIBLE_DEVICES=1 python main.py -m lenet300 -d mnist -bs 256 -e 100 -lr 1e-3 --prune_criterion flip \
                --flip_threshold 1 --opt adam --logdir="baselines" --comment="lenet300 original hparams threshold=1 no noise no wdecay anneal"

# LeNet5
CUDA_VISIBLE_DEVICES=1 python main.py -m lenet5 -d mnist -bs 256 -e 100 -lr 2e-3 --prune_criterion flip \
                --flip_threshold 1 --opt adam --logdir="baselines" --comment="lenet5 original hparams threshold=1 no noise no wdecay anneal"

# ResNet18
CUDA_VISIBLE_DEVICES=1 python main.py -m resnet18 -d cifar10 -bs 128 -e 350 -lr 0.1 --prune_criterion flip \
                --flip_threshold 1 --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 --logdir="baselines" --comment="resnet18 original hparams threshold=1 no noise no wdecay anneal"

# VGG19
CUDA_VISIBLE_DEVICES=1 python main.py -m vgg19 -d cifar10 -bs 128 -e 350 -lr 0.1 --prune_criterion flip \
                --flip_threshold 1 --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                 --milestones 150 250 --logdir="baselines" --comment="vgg19 original hparams threshold=1 no noise no wdecay anneal"
