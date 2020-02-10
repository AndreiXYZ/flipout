import numpy as np
import math, argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import json
import random

from utils import *
from data_loaders import *
from master_model import MasterWrapper
from snip import SNIP, apply_prune_mask
from epoch_funcs import *

def train(config, writer):
    device = config['device']
    model = load_model(config)

    # Send model to gpu and parallelize
    model = model.to(device)
    # model = nn.DataParallel(model)
    # Get train and test loaders
    train_loader, test_loader = load_dataset(config)

    train_size, test_size = len(train_loader.dataset), len(test_loader.dataset)
    
    opt = get_opt(config, model)
    epoch = get_epoch_type(config)
    # Do SNIP if it is the case
    if config['prune_criterion'] == 'snip':
        keep_percentage = 1 - config['snip_sparsity']
        keep_masks = SNIP(model, keep_percentage, train_loader, device)
        apply_prune_mask(model, keep_masks)
    else:
        model = MasterWrapper(model)
    
    print('Model has {} total params, including biases.'.format(model.get_total_params()))
    
    for epoch_num in range(1, config['epochs']+1):
        print('='*10 + ' Epoch ' + str(epoch_num) + ' ' + '='*10)
        
        model.train()
        # Anneal wdecay
        if config['anneal_lambda'] == True:
            opt.param_groups[0]['weight_decay'] = config['lambda']*(1-model.get_sparsity(config))

        
        train_acc, train_loss = epoch(epoch_num, train_loader, train_size, model, opt, writer, config)
        
        model.eval()
        with torch.no_grad():
            test_acc, test_loss = epoch(epoch_num, test_loader, test_size, model, opt, writer, config)
        
        if epoch_num%config['prune_freq'] == 0:
            if config['prune_criterion'] == 'magnitude':
                model.update_mask_magnitudes(config['prune_rate'])
            elif config['prune_criterion'] == 'flip':
                model.update_mask_flips(config['flip_threshold'])
            elif config['prune_criterion'] == 'random':
                model.update_mask_random(config['prune_rate'], config)
        
        print('Train - acc: {:>15.6f} loss: {:>15.6f}\nTest - acc: {:>16.6f} loss: {:>15.6f}'.format(
            train_acc, train_loss, test_acc, test_loss
        ))
        
        print('Sparsity : {:>15.4f}'.format(model.get_sparsity(config)))
        print('Wdecay : {:>15.6f}'.format(opt.param_groups[0]['weight_decay']))
        plot_stats(train_acc, train_loss, test_acc, test_loss, model, writer, epoch_num, config)

def main():
    config = parse_args()
    # Ensure experiment is reproducible.
    # Results may vary across machines!
    set_seed(config['seed'])
    # Set comment to name and then add hparams to tensorboard text
    writer = SummaryWriter(log_dir='./runs/' + config['logdir'] + '/' + get_time_str() + ' ' + config['comment'])
    del config['comment']
    writer.add_text('config', json.dumps(config))
    train(config, writer)

def parse_args():
    model_choices = ['lenet300', 'lenet5', 'conv6', 'vgg19', 'resnet18']
    pruning_choices = ['magnitude', 'flip', 'random', 'snip', 'l0']
    dataset_choices = ['mnist', 'cifar10']
    opt_choices = ['sgd', 'rmsprop', 'adam', 'rmspropw']

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, choices=model_choices, default='lenet300')
    parser.add_argument('-d', '--dataset', type=str, choices=['mnist', 'cifar10'], default='mnist')
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    # Pruning
    parser.add_argument('--prune_criterion', type=str, choices=pruning_choices, default=None)
    parser.add_argument('--prune_freq', type=int, default=1)
    parser.add_argument('--prune_rate', type=float, default=0.2) # for magnitude pruning
    parser.add_argument('--flip_threshold', type=int, default=1) # for flip pruning
    # Tensorboard-related args
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to add to tensorboard text')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Log dir. for tensorboard')
    # Optimizer args
    parser.add_argument('--opt', type=str, choices=['sgd', 'rmsprop', 'adam', 'rmspropw'])
    parser.add_argument('--momentum', 'mom', type=float, default=0)
    parser.add_argument('--reg_type', type=str, choices=['wdecay', 'l1', 'l2'])
    parser.add_argument('--lambda', type=float, default=0)
    parser.add_argument('--anneal_lambda', dest='anneal_lambda', action='store_true')
    parser.add_argument('--no_anneal_lambda', dest='anneal_lambda', action='store_false')
    # Add noise or not
    parser.add_argument('--noise', dest='add_noise', action='store_true')
    parser.add_argument('--no_noise', dest='add_noise', action='store_false')
    # SNIP params
    parser.add_argument('--snip_sparsity', type=float, required=False, default=0.)
    # L0 params
    # parser.add_argument('--beta_ema', type=float, default=0.999)
    # parser.add_argument('--lambas', nargs='*', type=float, default=[1., 1., 1., 1.])
    # parser.add_argument('--local_rep', action='store_true')
    # parser.add_argument('--temperature', type=float, default=2./3.)
    
    # Args for saving and resuming model training
    # parser.add_arugment('--save_model', action='store_true', default=False)
    # parser.add_argument('--resume', action='store_true', default=False)

    config = vars(parser.parse_args())
    
    return config

if __name__ == "__main__":
    main()