CUDA_VISIBLE_DEVICES=1 python main.py -m lenet300 -d mnist -bs 256 -e 10 -lr 1e-3 --prune_criterion topflip \
                --use_ema_flips --beta_ema_flips 0.5 --prune_freq 8 --prune_rate 0.2 --opt adam --noise \
                --reg_type wdecay --lambda 3e-3 \
                --logdir="test" --comment="sanity check"