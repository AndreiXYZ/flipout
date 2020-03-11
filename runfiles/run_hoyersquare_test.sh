CUDA_VISIBLE_DEVICES=1 python main.py -m lenet300 -d mnist -bs 256 -tbs 10000 -e 10 -lr 1e-3  --opt adam \
                    --prune_criterion sensitivity --prune_freq 10000 --sensitivity 0.03 --load_model chkpts/[11-03-20_14:55:20]lenet300_hoyer_square.pt\
                    --logdir="hoyer_square_tests" --comment="lenet300 hoyer square"

# CUDA_VISIBLE_DEVICES=1 python main.py -m lenet5 -d mnist -bs 256 -tbs 10000 -e 250 -lr 2e-3 --opt adam \
#                     --prune_criterion sensitivity --prune_freq 100 --sensitivity 0.03 --reg_type hs --lambda 1e-4 \
#                     --logdir="hoyer_square_tests" --comment="lenet5 hoyer square"