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