seed=44
device=3

for prune_freq in 117 70 50 39; do
    # Do magnitude and random
    for prune_criterion in 'magnitude' 'random';do
        # ResNet18
        CUDA_VISIBLE_DEVICES=${device} python main.py -m resnet18 -d cifar10 -bs 128 -tbs 10000 -e 350 -lr 0.1 \
                        --prune_criterion ${prune_criterion}  --prune_rate 0.5 --prune_freq  ${prune_freq} --seed ${seed} \
                        --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                        --milestones 150 250 --logdir="criterion_experiment" \
                        --comment="resnet18 crit=${prune_criterion} pf=${prune_freq} seed=${seed}" \
                        --save_model "resnet18_${prune_criterion}_pf${prune_freq}_s${seed}"

    done
    
    # Do flips
    CUDA_VISIBLE_DEVICES=${device} python main.py -m resnet18 -d cifar10 -bs 128 -tbs 10000 -e 350 -lr 0.1  \
                --prune_criterion topflip --prune_rate 0.5 --prune_freq ${prune_freq} --seed ${seed} \
                --opt sgd --momentum 0.9 --noise --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 --logdir="criterion_experiment" \
                --comment="resnet18 crit=topflip pf=${prune_freq} seed=${seed}" \
                --save_model "resnet18_topflip_pf${prune_freq}_s${seed}"

done