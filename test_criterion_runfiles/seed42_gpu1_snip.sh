#!/bin/bash
device=1
seed=42

sparsities=(
    `python -c "print(1-1/2**2)"`
    `python -c "print(1-1/2**4)"`
    `python -c "print(1-1/2**6)"` 
    `python -c "print(1-1/2**8)"`
)

for sparsity in ${sparsities[@]};do
    # ResNet18
    CUDA_VISIBLE_DEVICES=${device} python main.py -m resnet18 -d cifar10 -bs 128 -e 350 -tbs 10000 -lr 0.1 \
                    --prune_criterion snip --snip_sparsity ${sparsity} --seed ${seed} \
                    --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                    --milestones 150 250 --logdir="criterion_experiment_no_bias" \
                    --comment="resnet18 crit=snip sparsity=${sparsity} seed=${seed}" \
                    --save_model "pre-finetune/resnet18_snip_sp${sparsity}_s${seed}"
        
done

