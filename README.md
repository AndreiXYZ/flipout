## FlipOut : Uncovering redundant weights via sign-flipping

This is a repository of the code used to generate the experiments for [*FlipOut: Uncovering Redundant Weights via Sign Flipping*](https://arxiv.org/pdf/2009.02594.pdf). It contains the implementation of our proposed method as well as for the baselines. 

      * [FlipOut : Uncovering redundant weights via sign-flipping](#flipout--uncovering-redundant-weights-via-sign-flipping)
         * [Results](#results)
         * [Setup](#setup)
         * [Reproduce experiments](#reproduce-experiments)
         
### Results
We compare our method to:
- global magnitude pruning ([magnitude pruning](https://arxiv.org/abs/1506.02626) modified to rank weights globally according to the observations by [Frankle & Carbin](https://arxiv.org/abs/1803.03635))
- [SNIP](https://arxiv.org/abs/1810.02340) 
- [Hoyer-Square](https://openreview.net/pdf?id=rylBK34FDS).
- random pruning

Below, we provide the results from the paper, comparing our method to the baselines. Results are averaged over 3 runs; standard deviations are also included.

**VGG-19 on CIFAR10:**

![vgg19](/imgs/vgg19_results.png)

| Method | Sparsity (%)| Accuracy (%)|
| --- | --- | --- |
| FlipOut | 99.9 | **87.39 ± 0.23** |
| GlobalMagnitude | 99.9 | 82.89 ± 2.00 |
| Random | 99.9 | 10 ± 0 |
| SNIP | 99.9 | 10 ± 0 |
| Hoyer-Square (λ=6e-5) | 99.89 | 82.78 |

**ResNet18 on CIFAR10:**

![rn18](/imgs/resnet18_results.png)

| Method | Sparsity (%)| Accuracy (%)|
| --- | --- | --- |
| FlipOut | 99.9 | **82.5 ± 0.11** |
| GlobalMagnitude | 99.9 | 80.63 ± 0.45 |
| Random | 99.9 | 13.65 ± 6.32 |
| SNIP | 99.9 | 10 ± 0 |
| Hoyer-Square (λ=1e-4) | 99.89 | 78.58 |

**DesneNet-121 on Imagenette:**

![densenet121](/imgs/densenet121_results.png)

| Method | Sparsity (%)| Accuracy (%)|
| --- | --- | --- |
| FlipOut | 99.9 | 74.13 ± 1.4 |
| GlobalMagnitude | 99.9 | 67.94 ± 2.68 |
| Random | 99.9 | 9.9 ± 0.79 |
| SNIP | 99.9 | 9.09 ± 0 |
| Hoyer-Square (λ=3e-4) | 99.95 | **78.44** |

### Setup
Create a Conda virtual environment from ```environment.yml``` as follows:
```
conda env create -f environment.yml
source activate flipout
```
Then simply run:
```
./get_imagenette.sh
```
which will download the Imagenette dataset into a newly created ```data``` folder. For CIFAR10 or MNIST, PyTorch will automatically handle the download when first running a script.

### Reproduce experiments
Some of the methods perform pruning periodically and can have their final sparsity determined by the pruning rate (how many parameters are removed each time, in percentages) and frequency (how often we prune, in epochs). Below we include a reference table for the sparsities we used in our experiments and the pruning frequencies, assuming 350 epochs of training and a pruning rate of 50%.

| Sparsity | Prune frequency |
| --- | --- |
| 75% | 117 |
| 93.75% | 70 |
| 98.44% | 50 |
| 99.61% | 39 |
| 99.9% | 32 |

For SNIP, the sparsity can be directly selected. For Hoyer-Square, it is a function of the regularization term as well as the pruning threshold. 

Following you can find example commands to replicate the results on VGG19. For other model/dataset combinations, please skip to the last two paragraphs of the file. Note that random seeds do not transfer across machines, so your results may slightly differ.

**FlipOut (@99.9%, λ=1) :**
```
python main.py --model vgg19 --dataset cifar10 -bs 128 -e 350 -lr 0.1 \
                --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 \
                --prune_criterion flipout --flipout_p 2 \
                --prune_freq 32 --prune_rate 0.5 \
                --noise --noise_scale_factor 1 \
                --comment="test flipout" \
                --logdir=vgg19/
```
**Global Magnitude (@99.9%):**
```
python main.py --model vgg19 --dataset cifar10 -bs 128 -e 350 -lr 0.1 \
                --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 \
                --prune_criterion global_magnitude --prune_freq 32 --prune_rate 0.5 \
                --comment="test global magnitude" \
                --logdir=vgg19/
```
**Random (@99.9%):**
```
python main.py --model vgg19 --dataset cifar10 -bs 128 -e 350 -lr 0.1 \
                --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 \
                --prune_criterion random --prune_freq 32 --prune_rate 0.5 \
                --comment="test random" \
                --logdir=vgg19/
```
**SNIP (@99.9%)**:
```
python main.py --model vgg19 --dataset cifar10 -bs 128 -e 350 -lr 0.1 \
                --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 \
                --prune_criterion snip --snip_sparsity 0.999 \
                --comment="test snip" \
                --logdir=vgg19/
```
**Hoyer-Square (λ=6e-5, threshold=1e-4 )**:
```
python main.py --model vgg19 --dataset cifar10 -bs 128 -e 500 -lr 0.1 \
                --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 \
                --add_hs --hoyer_lambda 6e-5 \
                --prune_criterion threshold --magnitude_threshold 1e-4 \
                --prune_freq 350 --stop_hoyer_at 350 \
                --comment="test hoyer-square" \
                --logdir=vgg19/
```

The runs are saved in the directory specified by ```logdir``` with the filename ```comment``` and can be inspected with Tensorboard.

To run on different model/dataset combinations, simply replace the ```-m``` and ```-d``` arguments, i.e. ```-m vgg19 -d cifar10``` or ```-m densenet121 -d imagenette```. For other levels of sparsity, modify the prune frequency (```prune_freq```) according to the table at the top of the page, or the ```hoyer_lambda``` & ```magnitude_threshold``` parameters in Hoyer-Square and ```snip_sparsity``` in SNIP.
