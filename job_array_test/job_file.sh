#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --mem=16000M
#SBATCH --cpus-per-task=2
#SBATCH --array=1-4
#SBATCH --job-name=array_test
#SBATCH --output=array_test%a.out
source activate base

echo ${SLURM_ARRAY_TASK_ID}
param_folder='job_array_test/option_file'
echo `sed -n ${SLURM_ARRAY_TASK_ID}p ${param_folder}`

CUDA_VISIBLE_DEVICES=${device} python main.py -m vgg19 -d cifar10 -bs 128 -tbs 2000 -e 350 -lr 0.1 \
                --prune_criterion ${prune_criterion}  --prune_rate 0.5 --prune_freq  ${prune_freq} --seed ${seed} \
                --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 --logdir="criterion_experiment_no_bias/vgg19" \
                --comment="vgg19 crit=${prune_criterion} pf=${prune_freq} seed=${seed}" \
                --save_model "pre-finetune/vgg19_${prune_criterion}_pf${prune_freq}_s${seed}"