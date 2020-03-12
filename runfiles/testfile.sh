CUDA_VISIBLE_DEVICES=1 python main.py -m lenet300 -d mnist -bs 100 -tbs 10000 -e 10 -lr 1e-3  --opt adam \
                    --prune_criterion random --prune_freq 5  --prune_rate 0.5 --save_model \
                    --logdir="test" --comment="lenet300 pretrained"