# This file runs L0 (weight) regularization for LeNet300 and LeNet5

CUDA_VISIBLE_DEVICES=1 python main.py -m l0lenet5 -d mnist -bs 200 -e 200 -lr 1e-3 --opt adam \
                    --prune_criterion l0 --lambda 5e-4 --beta_ema 0.999 --lambas 50 50 50 50 50 \
                    --logdir="l0_tests" --comment="test lenet5 l0 lambas=50"

CUDA_VISIBLE_DEVICES=1 python main.py -m l0lenet300 -d mnist -bs 200 -e 200 -lr 1e-3 --opt adam \
                    --prune_criterion l0 --lambda 5e-4 --beta_ema 0.999 --lambas 50 50 50  \
                    --logdir="l0_tests" --comment="test lenet300 l0 lambas=50"

CUDA_VISIBLE_DEVICES=1 python main.py -m l0lenet5 -d mnist -bs 200 -e 200 -lr 1e-3 --opt adam \
                    --prune_criterion l0 --lambda 5e-4 --beta_ema 0.999 --lambas 25 25 25 25 25 \
                    --logdir="l0_tests" --comment="test lenet5 l0 lambas=25"

CUDA_VISIBLE_DEVICES=1 python main.py -m l0lenet300 -d mnist -bs 200 -e 200 -lr 1e-3 --opt adam \
                    --prune_criterion l0 --lambda 5e-4 --beta_ema 0.999 --lambas 25 25 25  \
                    --logdir="l0_tests" --comment="test lenet300 l0 lambas=25"