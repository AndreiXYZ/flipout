#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --partition=gpu_titanrtx_shared
#SBATCH --gres=gpu:1
#SBATCH --mem=16000M
#SBATCH --cpus-per-task=1
#SBATCH --array=1-30%15
#SBATCH --job-name=from_pretrained
#SBATCH --output=out_files/from_pretrained/array_job%a.out
source activate base
device=0;

param_folder='runfiles/args_from_pretrained';
run_params=`sed -n ${SLURM_ARRAY_TASK_ID}p ${param_folder}`;
echo Running ${run_params};

# When doing run from pretrained, we do not use scheduler and instead 
# use the latest learning rate
CUDA_VISIBLE_DEVICES=${device} python main.py -bs 128 -e 350 -lr 0.001 \
                --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 \
                ${run_params}
