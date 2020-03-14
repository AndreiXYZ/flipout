CUDA_VISIBLE_DEVICES=0 python main.py -m lenet300 -d mnist -bs 100 -tbs 10000 -e 30 -lr 1e-3  --opt adam \
                    --prune_criterion global_magnitude --prune_freq 2  --prune_rate 0.5 \
                    --logdir="test" --comment="lenet300 pretrained"