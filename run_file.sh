#!/bin/bash
. /etc/bashrc
. /etc/profile.d/modules.sh

module load cuda10.0/toolkit/10.0.130

echo "Running srun cmd with params:"
echo $@
echo $SLURM_LAUNCH_NODE_IPADDR 
srun --time=48:00:00 --mem=40GB -C TitanX --gres=gpu:1 python -u "$@"
