import numpy as np
import math, argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import json

import utils.utils as utils
import utils.plotters as plotters
import utils.getters as getters

from torch.utils.data import Subset
from utils.data_loaders import *
from torch.utils.tensorboard import SummaryWriter
from models.master_model import init_attrs, CustomDataParallel
from models.L0_models import L0MLP, L0LeNet5
from snip import SNIP, apply_prune_mask
from utils.calc_flops import get_flops

def train(config, writer):
    device = config['device']
    model = getters.get_model(config)
    # Send model to gpu and parallelize
    model = model.to(device)
    # model = nn.DataParallel(model)
    # Get train and test loaders
    train_loader, test_loader = getters.get_dataloaders(config)
    # Create a subset of a single sample for the FLOP calculation
    subset_train = Subset(train_loader.dataset, [0])
    mb_x, mb_y = next(iter(subset_train))
    mb_x = mb_x.unsqueeze(0)
    mb_x = mb_x.to(config['device'])

    train_dataset_size, test_dataset_size = len(train_loader.dataset), len(test_loader.dataset)

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
    if config['load_model'] is not None:
        checkpoint = torch.load(config['load_model'], map_location='cuda')
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
        train_acc, train_loss = epoch(epoch_num, train_loader, train_dataset_size, model, opt, writer, config)
        
        model.eval()
        with torch.no_grad():
            test_acc, test_loss = epoch(epoch_num, test_loader, test_dataset_size, model, opt, writer, config)

        if config['use_scheduler']:
            scheduler.step()

        # Prune only if stop_pruning_at is not set or the current epoch is lower than the stopping point
        if config['stop_pruning_at'] == -1 or epoch_num < config['stop_pruning_at']:
            if epoch_num%config['prune_freq'] == 0 and epoch_num != config['epochs']:
                if config['prune_criterion'] == 'magnitude':
                    model.update_mask_magnitudes(config['prune_rate'])
                elif config['prune_criterion'] == 'flip':
                    model.update_mask_flips(config['flip_threshold'])
                elif config['prune_criterion'] == 'topflip':
                    model.update_mask_topflips(config['prune_rate'], config['use_ema_flips'], config['reset_flip_cts'])
                elif config['prune_criterion'] == 'topflip_layer':
                    model.update_mask_topflips_layerwise(config['prune_rate'])
                elif config['prune_criterion'] == 'random':
                    model.update_mask_random(config['prune_rate'], config)
                elif config['prune_criterion'] == 'sensitivity':
                    model.update_mask_sensitivity(config['sensitivity'])
                elif  config['prune_criterion'] == 'global_magnitude':
                    model.update_mask_global_magnitudes(config['prune_rate'])
                elif config['prune_criterion'] == 'historical_magnitude':
                    model.update_mask_historical_magnitudes(config['prune_rate'])
                elif config['prune_criterion'] == 'weight_div_flips':
                    model.update_mask_weight_div_flips(config['prune_rate'])
                elif config['prune_criterion'] == 'weight_squared_div_flips':
                    model.update_mask_weight_squared_div_flips(config['prune_rate'])
                elif config['prune_criterion'] == 'weight_div_squared_flips':
                    model.update_mask_weight_div_squared_flips(config['prune_rate'])
                elif config['prune_criterion'] == 'threshold':
                    model.update_mask_threshold(config['magnitude_threshold'])
                elif config['prune_criterion'] == 'structured_magnitude':
                    model.update_mask_structured_magnitudes(config)
                # Always also print the nonzeros to see which layers get pruned
                if config['prune_criterion'] != 'none':
                    utils.print_nonzeros(model)

                # Plot layerwise sparsity
                # plotters.plot_layerwise_sparsity(model, writer, epoch_num)
        
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
        
        # Grab single mini-batch for FLOPs calculation
        
        total_flops, nonzero_flops = get_flops(model, mb_x)

        print('#FLOPs : total={} nonzero={} reduction rate={}'.format(
            total_flops, nonzero_flops, float(total_flops)/nonzero_flops
        ))
        
        plotters.plot_stats(train_acc, train_loss, test_acc, test_loss, 
                    model, writer, epoch_num, config, cls_module)

        
    # After training is done, log the hparams and the metrics
    # plot_hparams(writer, config, train_acc, test_acc, train_loss, test_loss, model.sparsity)
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
        utils.save_run(model, opt, config)


def parse_args():
    model_choices = ['lenet300', 'lenet5', 'conv6', 'vgg19', 'resnet18',
                     'l0lenet5', 'l0lenet300', 'densenet121']
    
    pruning_choices = ['magnitude', 'flip', 'topflip', 'topflip_layer', 
                       'random', 'snip', 'l0', 'none', 'sensitivity',
                       'global_magnitude', 'historical_magnitude',
                       'weight_div_flips', 'weight_squared_div_flips',
                       'weight_div_squared_flips', 'threshold',
                       'structured_magnitude']
    
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

    # Pruning
    parser.add_argument('--prune_criterion', type=str, choices=pruning_choices, default='none')
    parser.add_argument('--prune_freq', type=int, default=1)
    parser.add_argument('--prune_rate', type=float, default=0.2) # for magnitude pruning
    parser.add_argument('--sensitivity', type=float, default=0)
    parser.add_argument('--flip_threshold', type=int, default=1) # for flip pruning
    parser.add_argument('--magnitude_threshold', type=float, default=0)
    parser.add_argument('--stop_pruning_at', type=int, default=-1)
    parser.add_argument('--prune_bias', action='store_true', default=False)
    parser.add_argument('--prune_bnorm', action='store_true', default=False)
    # Deep Hoyer regularizer
    parser.add_argument('--add_ghs', action='store_true', default=False)
    parser.add_argument('--add_hs', action='store_true', default=False)
    parser.add_argument('--hoyer_lambda', type=float, default=0)
    parser.add_argument('--stop_hoyer_at', type=int, default=-1)
    # Flip pruning EMA
    parser.add_argument('--use_ema_flips', dest='use_ema_flips', action='store_true', default=False)
    parser.add_argument('--beta_ema_flips', type=float, default=None)
    parser.add_argument('--reset_flip_cts', action='store_true', default=False)
    # Historical magnitudes params
    parser.add_argument('--normalize_magnitudes', action='store_true', default=False)
    parser.add_argument('--beta_ema_maghists', type=float, default=None)
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
    parser.add_argument('--scale_noise_by_lr', dest='scale_noise_by_lr', action='store_true', default=False)
    parser.add_argument('--stop_noise_at', type=int, default=-1)
    parser.add_argument('--noise_only_prunable', action='store_true', default=False)
    parser.add_argument('--noise_scale_factor', type=float, default=1)
    parser.add_argument('--global_noise', action='store_true', default=False)
    # SNIP params
    parser.add_argument('--snip_sparsity', type=float, required=False, default=0.)
    # L0 params
    parser.add_argument('--beta_ema', type=float, default=0.999)
    parser.add_argument('--lambas', nargs='*', type=float, default=[1., 1., 1., 1.])
    parser.add_argument('--local_rep', action='store_true')
    parser.add_argument('--temperature', type=float, default=2./3.)
    
    # Whether or not to save the model. Run-name will be comment name
    parser.add_argument('--save_model', type=str, default=None)
    parser.add_argument('--load_model', type=str, default=None)
    # Arg for parallelizing everything
    parser.add_argument('--parallel', action='store_true', default=False)
    config = vars(parser.parse_args())
    
    return config

if __name__ == "__main__":
    main()