import numpy as np
import math, argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import json

import utils.utils as utils
import utils.getters as getters

from torch.utils.data import Subset
from utils.data_loaders import *
from torch.utils.tensorboard import SummaryWriter
from models.master_model import init_attrs, CustomDataParallel
from snip import SNIP, apply_prune_mask
from utils.calc_flops import get_flops

def train(config, writer):
    device = config['device']
    model = getters.get_model(config)
    # Send model to gpu and parallelize
    model = model.to(device)
    # model = nn.DataParallel(model)
    # Get train and test loaders
    loaders, sizes = getters.get_dataloaders(config)
    train_loader, val_loader, test_loader = loaders
    train_size, val_size, test_size = sizes
    
    # Create a subset of a single sample for the FLOP calculation
    subset_train = Subset(train_loader.dataset, [0])
    mb_x, mb_y = next(iter(subset_train))
    mb_x = mb_x.unsqueeze(0)
    mb_x = mb_x.to(config['device'])


    opt = getters.get_opt(config, model)
    epoch = getters.get_epoch_type(config)

    if config['use_scheduler']:
        scheduler = lr_scheduler.MultiStepLR(opt, milestones=config['milestones'], gamma=0.1)
    
    # Do SNIP if it is the case
    if config['prune_criterion'] == 'snip':
        keep_percentage = 1 - config['snip_sparsity']
        keep_masks = SNIP(model, keep_percentage, train_loader, device)
        apply_prune_mask(model, keep_masks)
        init_attrs(model, config)
        model.sparsity = model.get_sparsity(config)
    else:
        init_attrs(model, config)

    # Init the mask from the loader if it is the case
    if config['load_model']:
        raise NotImplementedError
        checkpoint = torch.load(config['load_model'], map_location='cuda')
        model.load_state_dict(checkpoint['model_state'])
        opt.load_state_dict(checkpoint['opt_state'])
        model.mask = checkpoint['mask']
    
    if config['parallel']:
        model = CustomDataParallel(model)
    
    # Grab the final classification layer to check disconnects
    modules = [module for module in model.modules()
                if hasattr(module, 'weight')]
    cls_module = modules[-1]

    num_weights, num_biases = model.get_total_params()

    print('Model has {} total params.\nnum_weights={}\nnum_biases={}\nnum.prunable={}'
          .format(num_weights+num_biases, num_weights, num_biases,
          model.total_prunable)
          )
    
    if not config['prune_bias']:
        print('---Biases omitted from pruning---')
    if not config['prune_bnorm']:
        print('---Bnorm omitted from pruning---')
    
    for epoch_num in range(1, config['epochs']+1):
        print('='*10 + ' Epoch ' + str(epoch_num) + ' ' + '='*10)
        
        model.train()
        # Anneal wdecay
        train_acc, train_loss = epoch(epoch_num, train_loader, train_size, model, opt, writer, config)
        
        model.eval()
        with torch.no_grad():
            if config['evaluate_on_val']:
                test_acc, test_loss = epoch(epoch_num, val_loader, val_size, model, opt, writer, config)
            else:
                test_acc, test_loss = epoch(epoch_num, test_loader, test_size, model, opt, writer, config)

        if config['use_scheduler']:
            scheduler.step()

        # Prune only if stop_pruning_at is not set or the current epoch is 
        # lower than the stopping point
        if config['stop_pruning_at'] == -1 or epoch_num < config['stop_pruning_at']:
            if epoch_num%config['prune_freq'] == 0 and epoch_num != config['epochs']:
                
                if config['prune_criterion'] == 'magnitude':
                    model.update_mask_magnitudes(config['prune_rate'])

                elif config['prune_criterion'] == 'random':
                    model.update_mask_random(config['prune_rate'], config)

                elif  config['prune_criterion'] == 'global_magnitude':
                    model.update_mask_global_magnitudes(config['prune_rate'])

                elif config['prune_criterion'] == 'flipout':
                    model.update_mask_flipout(config['prune_rate'], config['flipout_p'])

                elif config['prune_criterion'] == 'threshold':
                    model.update_mask_threshold(config['magnitude_threshold'])
                # Always also print the nonzeros to see which layers get pruned
                if config['prune_criterion'] != 'none':
                    utils.print_nonzeros(model)
        
        # Update model's sparsity
        model.sparsity = model.get_sparsity(config)
        
        if config['anneal_lambda'] == True:
            opt.param_groups[0]['weight_decay'] = config['lambda']*(1-model.sparsity)

        if config['anneal_lr'] == True:
            opt.param_groups[0]['lr'] = config['lr']*(1-model.sparsity)
        
        print('LR = ', opt.param_groups[0]['lr'])
        
        print('Train - acc: {:>15.6f} loss: {:>15.6f}\nTest - acc: {:>16.6f} loss: {:>15.6f}'.format(
            train_acc, train_loss, test_acc, test_loss
        ))
        
        print('Sparsity : {:>15.4f}'.format(model.sparsity))
        print('Wdecay : {:>15.6f}'.format(opt.param_groups[0]['weight_decay']))

        utils.plot_stats(train_acc, train_loss, test_acc, test_loss, 
                    model, writer, epoch_num, config)
        
    return model, opt

def main():
    config = parse_args()
    # Ensure experiment is reproducible.
    # Results may vary across machines!
    utils.set_seed(config['seed'])
    # Set comment to name and then add hparams to tensorboard text
    curr_time = utils.get_time_str()
    logdir = './runs/' + config['logdir'] + '/' + curr_time + ' ' + config['comment']
    writer = SummaryWriter(log_dir=logdir)

    comment = config.pop('comment')
    writer.add_text('config', json.dumps(config, indent=4))

    print('*'*30 + '\nRunning\n' + json.dumps(config, indent=4) + '\n' + '*'*30)
    
    model, opt = train(config, writer)
    
    writer.flush()
    writer.close()

    utils.print_nonzeros(model)
    
    if config['save_model'] is not None:
        utils.save_run(model, opt, config, logdir)


def parse_args():
    model_choices = ['lenet300', 'lenet5', 'conv6', 'vgg19', 'vgg16', 'vgg13',
                     'resnet18', 'densenet121']
    
    pruning_choices = ['magnitude', 'random', 'snip', 'none', 
                       'global_magnitude', 'flipout', 'threshold']
    
    opt_choices = ['sgd', 'rmsprop', 'adam']
    reg_type_choices = ['wdecay', 'l1', 'l2']
    dataset_choices = ['mnist', 'cifar10', 'imagenette']

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, choices=model_choices, default='lenet300')
    parser.add_argument('-d', '--dataset', type=str, choices=dataset_choices, default='mnist')
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-tbs', '--test_batch_size', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    # Args for val set
    parser.add_argument('--val', action='store_true', default=False)
    parser.add_argument('--val_size', type=int, default=0)
    parser.add_argument('--evaluate_on_val', action='store_true', default=False)
    # Pruning
    parser.add_argument('--prune_criterion', type=str, choices=pruning_choices, default='none')
    parser.add_argument('--prune_freq', type=int, default=1)
    parser.add_argument('--prune_rate', type=float, default=0.2)
    parser.add_argument('--magnitude_threshold', type=float, default=0)
    parser.add_argument('--stop_pruning_at', type=int, default=-1)
    parser.add_argument('--prune_bias', action='store_true', default=False)
    parser.add_argument('--prune_bnorm', action='store_true', default=False)
    # Flipout p
    parser.add_argument('--flipout_p', type=int, default=2)
    # Deep Hoyer regularizer
    parser.add_argument('--add_hs', action='store_true', default=False)
    parser.add_argument('--hoyer_lambda', type=float, default=0)
    parser.add_argument('--stop_hoyer_at', type=int, default=-1)
    # Tensorboard-related args
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to add to tensorboard text')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Log dir. for tensorboard')
    # Optimizer args
    parser.add_argument('--opt', type=str, choices=opt_choices)
    parser.add_argument('--momentum', '-mom', type=float, default=0)
    parser.add_argument('--use_scheduler', dest='use_scheduler', action='store_true', default=False)
    parser.add_argument('--milestones', nargs='*', type=int, required=False)
    parser.add_argument('--reg_type', type=str, choices=reg_type_choices, default=None)
    parser.add_argument('--lambda', type=float, default=0)
    parser.add_argument('--anneal_lambda', dest='anneal_lambda', action='store_true', default=False)
    parser.add_argument('--anneal_lr', dest='anneal_lr', action='store_true', default=False)
    # Params for gradient norm clipping
    parser.add_argument('--clip_grad', action='store_true', default=False)
    parser.add_argument('--max_norm', type=float, default=None)
    # Add noise or not
    parser.add_argument('--noise', dest='add_noise', action='store_true', default=False)
    parser.add_argument('--stop_noise_at', type=int, default=-1)
    parser.add_argument('--noise_only_prunable', action='store_true', default=False)
    parser.add_argument('--noise_scale_factor', type=float, default=1)
    # SNIP params
    parser.add_argument('--snip_sparsity', type=float, required=False, default=0.)
    # Whether or not to save the model. Run-name will be comment name
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--load_model', action='store_true', default=False)
    parser.add_argument('--model_load_location', type=str, default=None)
    # Arg for parallelizing everything
    parser.add_argument('--parallel', action='store_true', default=False)
    config = vars(parser.parse_args())
    
    return config

if __name__ == "__main__":
    main()