for seed in 42 43 44; do
    for prune_freq in 117 70 50 39; do
        
        # Do magnitude and random
        for prune_criterion in 'magnitude' 'random';do
        
            # echo $seed $prune_criterion 
            echo "reset18 crit=$prune_criterion pf=$prune_freq seed=$seed"
            # ResNet18
            # CUDA_VISIBLE_DEVICES=1 python main.py -m resnet18 -d cifar10 -bs 128 -e 350 -lr 0.1 --prune_criterion $prune_criterion \
            #                 --prune_rate 0.5 --prune_freq  \
            #                 --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
            #                 --milestones 150 250 --logdir="baselines" --comment="resnet18 unpruned"

            # # VGG19
            # CUDA_VISIBLE_DEVICES=1 python main.py -m vgg19 -d cifar10 -bs 128 -e 350 -lr 0.1 --prune_criterion $prune_criterion \
            #                 --prune_rate 0.5 --prune_freq \
            #                 --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
            #                 --milestones 150 250 --logdir="baselines" --comment="vgg19 unpruned"
        done

        # Do flips
        ...
        # Do SNIP
    done
done