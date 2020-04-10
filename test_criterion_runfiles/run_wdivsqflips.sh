#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --mem=16000M
#SBATCH --cpus-per-task=1
#SBATCH --array=1-15
#SBATCH --job-name=wdivsqflips_resnet18
#SBATCH --output=out_files/wdivsqflips_resnet18/array_job%a.out
source activate base
device=0;

param_folder='test_criterion_runfiles/array_job_args';
run_params=`sed -n ${SLURM_ARRAY_TASK_ID}p ${param_folder}`;
echo Running ${run_params};

CUDA_VISIBLE_DEVICES=${device} python main.py -bs 128 -tbs 5000 -e 350 -lr 0.1 \
                --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 \
                ${run_params}
