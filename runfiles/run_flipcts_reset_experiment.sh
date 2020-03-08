
# for seed in 44; do
#     # ResNet18
#     CUDA_VISIBLE_DEVICES=1 python main.py -m resnet18 -d cifar10 -bs 128 -e 350 -lr 0.1 --prune_criterion topflip \
#                     --prune_freq 14 --prune_rate 0.2 --opt sgd --momentum 0.9 --noise --reg_type wdecay --lambda 5e-4 --use_scheduler \
#                     --seed $seed --reset_flip_cts --milestones 150 250 \
#                     --logdir="flipcts_test" --comment="resnet18 topflip flipcts_reset"

#     # VGG19
#     CUDA_VISIBLE_DEVICES=1 python main.py -m vgg19 -d cifar10 -bs 128 -e 350 -lr 0.1 --prune_criterion topflip \
#                     --prune_freq 14 --prune_rate 0.2 --opt sgd --momentum 0.9 --noise --reg_type wdecay --lambda 5e-4 --use_scheduler \
#                     --seed $seed --reset_flip_cts --milestones 150 250 \
#                     --logdir="flipcts_test" --comment="vgg19 topflip flipcts_reset"
# done



for seed in 42 43 44; do
    # LeNet300
    CUDA_VISIBLE_DEVICES=1 python main.py -m lenet300 -d mnist -bs 256 -e 200 -lr 1e-3 --prune_criterion topflip \
                    --use_ema_flips --beta_ema_flips 0.9 --prune_freq 8 --prune_rate 0.2 --opt adam --noise --seed $seed \
                    --logdir="flipcts_test" --comment="lenet300 topflip ema_flips beta=0.9"

    # LeNet5
    CUDA_VISIBLE_DEVICES=1 python main.py -m lenet5 -d mnist -bs 256 -e 200 -lr 2e-3 --prune_criterion topflip \
                    --use_ema_flips --beta_ema_flips 0.9 --prune_freq 8 --prune_rate 0.2 --opt adam --noise --seed $seed \
                    --logdir="flipcts_test" --comment="lenet5 topflip ema_flips beta=0.9"

    # ResNet18
    CUDA_VISIBLE_DEVICES=1 python main.py -m resnet18 -d cifar10 -bs 128 -e 350 -lr 0.1 --prune_criterion topflip \
                    --prune_freq 14 --prune_rate 0.2 --opt sgd --momentum 0.9 --noise --reg_type wdecay --lambda 5e-4 --use_scheduler \
                    --seed $seed --use_ema_flips --beta_ema_flips 0.9 --milestones 150 250 \
                    --logdir="flipcts_test" --comment="resnet18 topflip ema_flips beta=0.9"

    # VGG19
    CUDA_VISIBLE_DEVICES=1 python main.py -m vgg19 -d cifar10 -bs 128 -e 350 -lr 0.1 --prune_criterion topflip \
                    --prune_freq 14 --prune_rate 0.2 --opt sgd --momentum 0.9 --noise --reg_type wdecay --lambda 5e-4 --use_scheduler \
                    --seed $seed --use_ema_flips --beta_ema_flips 0.9 --milestones 150 250 \
                    --logdir="flipcts_test" --comment="vgg19 topflip ema_flips beta=0.9"
done


# for seed in 42 43 44; do
#     # LeNet300
#     CUDA_VISIBLE_DEVICES=1 python main.py -m lenet300 -d mnist -bs 256 -e 200 -lr 1e-3 --prune_criterion topflip \
#                     --prune_freq 8 --prune_rate 0.2 --opt adam --noise --seed $seed \
#                     --logdir="flipcts_test" --comment="lenet300 topflip just_add_flips"

#     # LeNet5
#     CUDA_VISIBLE_DEVICES=1 python main.py -m lenet5 -d mnist -bs 256 -e 200 -lr 2e-3 --prune_criterion topflip \
#                     --prune_freq 8 --prune_rate 0.2 --opt adam --noise --seed $seed \
#                     --logdir="flipcts_test" --comment="lenet5 topflip just_add_flips"

#     # ResNet18
#     CUDA_VISIBLE_DEVICES=1 python main.py -m resnet18 -d cifar10 -bs 128 -e 350 -lr 0.1 --prune_criterion topflip \
#                     --prune_freq 14 --prune_rate 0.2 --opt sgd --momentum 0.9 --noise --reg_type wdecay --lambda 5e-4 --use_scheduler \
#                     --seed $seed --milestones 150 250 \
#                     --logdir="flipcts_test" --comment="resnet18 topflip just_add_flips"

#     # VGG19
#     CUDA_VISIBLE_DEVICES=1 python main.py -m vgg19 -d cifar10 -bs 128 -e 350 -lr 0.1 --prune_criterion topflip \
#                     --prune_freq 14 --prune_rate 0.2 --opt sgd --momentum 0.9 --noise --reg_type wdecay --lambda 5e-4 --use_scheduler \
#                     --seed $seed --milestones 150 250 \
#                     --logdir="flipcts_test" --comment="vgg19 topflip just_add_flips"
# done
