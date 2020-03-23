#!/bin/bash
device=3
seed=44

sparsities=(
    `python -c "print(1-1/2**2)"`
    `python -c "print(1-1/2**4)"`
    `python -c "print(1-1/2**6)"` 
    `python -c "print(1-1/2**8)"`
)

for prune_freq in 117 70 50 39; do
    # Do magnitude and random
    for prune_criterion in 'global_magnitude' 'random';do
        # vgg19
        CUDA_VISIBLE_DEVICES=${device} python main.py -m vgg19 -d cifar10 -bs 128 -tbs 10000 -e 350 -lr 0.1 \
                        --prune_criterion ${prune_criterion}  --prune_rate 0.5 --prune_freq  ${prune_freq} --seed ${seed} \
                        --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                        --milestones 150 250 --logdir="criterion_experiment_no_bias" \
                        --comment="vgg19 crit=${prune_criterion} pf=${prune_freq} seed=${seed}" \
                        --save_model "pre-finetune/vgg19_${prune_criterion}_pf${prune_freq}_s${seed}"

    done
    
    # Do flips
    CUDA_VISIBLE_DEVICES=${device} python main.py -m vgg19 -d cifar10 -bs 128 -tbs 10000 -e 350 -lr 0.1  \
                --prune_criterion topflip --prune_rate 0.5 --prune_freq ${prune_freq} --seed ${seed} \
                --opt sgd --momentum 0.9 --noise --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 --logdir="criterion_experiment_no_bias" \
                --comment="vgg19 crit=topflip pf=${prune_freq} seed=${seed}" \
                --save_model "pre-finetune/vgg19_topflip_pf${prune_freq}_s${seed}"
    
done

# Do SNIP
for sparsity in ${sparsities[@]};do
# vgg19
    CUDA_VISIBLE_DEVICES=${device} python main.py -m vgg19 -d cifar10 -bs 128 -e 350 -tbs 10000 -lr 0.1 \
                    --prune_criterion snip --snip_sparsity ${sparsity} --seed ${seed} \
                    --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                    --milestones 150 250 --logdir="criterion_experiment_no_bias" \
                    --comment="vgg19 crit=snip sparsity=${sparsity} seed=${seed}" \
                    --save_model "pre-finetune/vgg19_snip_sp${sparsity}_s${seed}"
    
done