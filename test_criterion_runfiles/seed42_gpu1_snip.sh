sparsities = (python -c "print(1-1/2**2)"
              python -c "print(1-1/2**4)"
              python -c "print(1-1/2**6)"
              python -c "print(1-1/2**8)"
    )

for sparsity in sparsities;do
    echo $sparsity
done

# ResNet18
# CUDA_VISIBLE_DEVICES=1 python main.py -m resnet18 -d cifar10 -bs 128 -e 350 -lr 0.1 --prune_criterion snip \
#                 --snip_sparsity 0.95 --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
#                 --milestones 150 250 --logdir="baselines" --comment="resnet18 snip=0.95" \
