#!/bin/bash
#SBATCH -t 16:00:00
#SBATCH -N 1
#SBATCH -p gpu_shared
#SBATCH --mem=16000M
#SBATCH --gres=gpu:1
#SBATCH --output=out_files/historical_magnitude_normalized_ema
source activate base
device=0;
seed=42;
echo "Running historical magnitudes with normalization.";
for prune_freq in 50 39; do
    #Do flips while adding noise only to prunable params
    CUDA_VISIBLE_DEVICES=${device} python main.py -m resnet18 -d cifar10 -bs 128 -tbs 5000 -e 350 -lr 0.1  \
                --prune_criterion historical_magnitude --prune_rate 0.5 --prune_freq ${prune_freq} --seed ${seed} \
                --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 --logdir="historical_magnitude_test" \
                --comment="resnet18 crit=historical_magnitude pf=${prune_freq} seed=${seed} normalized+ema"\
                --normalize_magnitudes --beta_ema_maghists 0.9
done