
# The idea for this run is to do pre-training warmup to gather the statistics, prune the 
# weights in one-shot and then fine-tune without noise.
for seed in 42; do
    # ResNet18
    CUDA_VISIBLE_DEVICES=0 python main.py -m resnet18 -d cifar10 -bs 128 -e 500 -lr 0.1 --prune_criterion topflip \
                    --prune_freq 150 --prune_rate 0.995 --stop_pruning_at 151 --opt sgd --momentum 0.9 --noise --reg_type wdecay --lambda 5e-4 --use_scheduler \
                    --seed $seed --milestones 300 400 --stop_noise_at 150 \
                    --logdir="oneshot_test" --comment="resnet18 topflip oneshot+finetune"

    # VGG19
    CUDA_VISIBLE_DEVICES=0 python main.py -m vgg19 -d cifar10 -bs 128 -e 500 -lr 0.1 --prune_criterion topflip \
                    --prune_freq 150 --prune_rate 0.995 --stop_pruning_at 151 --opt sgd --momentum 0.9 --noise --reg_type wdecay --lambda 5e-4 --use_scheduler \
                    --seed $seed --milestones 300 400 --stop_noise_at 150 \
                    --logdir="oneshot_test" --comment="vgg19 topflip oneshot+finetune"

done