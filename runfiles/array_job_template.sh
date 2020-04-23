#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --partition=gpu_titanrtx_shared
#SBATCH --gres=gpu:1
#SBATCH --mem=16000M
#SBATCH --cpus-per-task=1
#SBATCH --array=1-30
#SBATCH --job-name=some_job_name
#SBATCH --output=out_files/some_output/array_job%a.out
source activate base
device=0;

param_folder='runfiles/some_arg_file';
run_params=`sed -n ${SLURM_ARRAY_TASK_ID}p ${param_folder}`;
echo Running ${run_params};

CUDA_VISIBLE_DEVICES=${device} python main.py -bs 128 -e 350 -lr 0.1 \
                --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 \
                ${run_params}
