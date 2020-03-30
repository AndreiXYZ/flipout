#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --mem=16000M
#SBATCH --cpus-per-task=1
#SBATCH --array=1-60%30
#SBATCH --job-name=array_job_vgg19
#SBATCH --output=out_files/array_job%a.out
source activate base
device=0;

param_folder='test_criterion_runfiles/array_job_args'
run_params=`sed -n ${SLURM_ARRAY_TASK_ID}p ${param_folder}`;
echo Running ${run_params}

CUDA_VISIBLE_DEVICES=${device} python main.py -m vgg19 -d cifar10 -bs 128 -tbs 2000 -e 350 -lr 0.1 \
                --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 --logdir="criterion_experiment_no_bias/vgg19" \
                ${run_params}
