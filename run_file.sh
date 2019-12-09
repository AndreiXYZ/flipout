#!/bin/bash

module load cuda10.0/toolkit/10.0.130
module load python/3.6.0

srun --time=48:00:00 --mem=24GB -C TitanX --gres=gpu:1 python -u "$@"
