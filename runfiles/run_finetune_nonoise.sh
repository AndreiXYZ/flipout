for seed in 42 43 44; do
    # ResNet18
    CUDA_VISIBLE_DEVICES=1 python main.py -m resnet18 -d cifar10 -bs 128 -e 500 -lr 0.1 --prune_criterion topflip \
                    --prune_freq 14 --prune_rate 0.2 --opt sgd --momentum 0.9 --noise --reg_type wdecay --lambda 5e-4 --use_scheduler \
                    --seed $seed --milestones 150 250 --stop_pruning_at 349 --stop_noise_at 350 \
                    --logdir="finetune_test" --comment="resnet18 topflip just_add + finetune"

    # VGG19
    CUDA_VISIBLE_DEVICES=1 python main.py -m vgg19 -d cifar10 -bs 128 -e 500 -lr 0.1 --prune_criterion topflip \
                    --prune_freq 14 --prune_rate 0.2 --opt sgd --momentum 0.9 --noise --reg_type wdecay --lambda 5e-4 --use_scheduler \
                    --seed $seed --milestones 150 250 --stop_pruning_at 349 --stop_noise_at 350 \
                    --logdir="finetune_test" --comment="vgg19 topflip just_add + finetune"
done