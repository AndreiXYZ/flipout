CUDA_VISIBLE_DEVICES=1 python main.py -m lenet300 -d mnist -bs 100 -tbs 10000 -e 250 -lr 1e-3  --opt adam \
                    --prune_criterion sensitivity --prune_freq 2 --sensitivity 0.03 --save_model \
                    --logdir="hoyer_square_tests" --comment="lenet300 pretrained"

CUDA_VISIBLE_DEVICES=1 python main.py -m lenet300 -d mnist -bs 100 -tbs 10000 -e 251 -lr 1e-3 --opt adam \
                    --prune_criterion sensitivity --prune_freq 250 --sensitivity 0.05 --reg_type hs --lambda 0.001 \
                    --load_model chkpts/lenet300_pretrained.pt --save_model \
                    --logdir="hoyer_square_tests" --comment="lenet300 hoyer square"

CUDA_VISIBLE_DEVICES=1 python main.py -m lenet5 -d mnist -bs 100 -tbs 10000 -e 250 -lr 1e-3 --opt adam \
                    --prune_criterion none --save_model \
                    --logdir="hoyer_square_tests" --comment="lenet5 pretrained"

CUDA_VISIBLE_DEVICES=1 python main.py -m lenet5 -d mnist -bs 100 -tbs 10000 -e 251 -lr 1e-3 --opt adam \
                    --prune_criterion sensitivity --prune_freq 250 --sensitivity 0.05 --reg_type hs --lambda 0.001 \
                    --load_model chkpts/lenet5_pretrained.pt --save_model \
                    --logdir="hoyer_square_tests" --comment="lenet5 hoyer square"