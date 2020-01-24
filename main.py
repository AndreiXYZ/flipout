import torch, argparse, sys
import numpy as np
import math
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import json

from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from utils import *
from models import *
from data_loaders import *
from master_model import MasterWrapper
from snip import SNIP, apply_prune_mask

grads_alive_mean = []
grads_alive_std = []
grads_pruned_mean = []
grads_pruned_std = []

def epoch(epoch_num, loader,  model, opt, writer, config):
    epoch_acc = 0
    epoch_loss = 0
    size = len(loader.dataset)

    global grads_alive_mean
    global grads_alive_std
    global grads_pruned_mean
    global grads_pruned_std

    for batch_num, (x,y) in enumerate(loader):
        update_num = epoch_num*size/math.ceil(config['batch_size']) + batch_num
        opt.zero_grad()
        x = x.float().to(config['device'])
        y = y.to(config['device'])
        out = model.forward(x)

        # Calc loss function

        sparsity = model.get_sparsity(config)
        weight_penalty = model.get_weight_penalty(config)
        if config['anneal_lambda'] == True:
            weight_penalty *= (1-sparsity)
        loss = F.cross_entropy(out, y) + weight_penalty*config['lambda']

        if model.training:   
            if 'custom' not in config['model']:
                model.save_weights()
            loss.backward()

            model.apply_mask(config)
            
            if config['add_noise']:
                noise_per_layer = model.inject_noise(config)

                for idx,(name,layer) in enumerate(model.named_parameters()):
                    if 'weight' in name and ('fc' in name or 'conv' in name):
                        writer.add_scalar('noise/'+str(idx), noise_per_layer[idx], update_num)
                    break
            opt.step()

            # Monitor wegiths for flips
            if 'custom' not in config['model']: 
                flips_since_last = model.store_flips_since_last()

            # flips_total = model.get_flips_total()
            # flipped_total = model.get_total_flipped()
            # writer.add_scalar('flips/flips_since_last', flips_since_last, update_num)
            # writer.add_scalar('flips/flipped_total', flipped_total, update_num)

        else:
            # Get the gradients, averaged over n, the batch size
            # Accumulate them for all mini-batches
            loss.backward()

        preds = out.argmax(dim=1, keepdim=True).squeeze()
        correct = preds.eq(y).sum().item()
        
        epoch_acc += correct
        epoch_loss += loss.item()
            # When we exit the batch iteration, the .grad param
        # will have the average gradient across all mini-batches
    if model.training is False:
        with torch.no_grad():
            grads_alive = []
            grads_pruned = []
            for layer in model.parameters():
                grads_alive.append(layer.grad[layer!=0.].abs())
                grads_pruned.append(layer.grad[layer==0.].abs())
            grads_alive = torch.cat(grads_alive)
            grads_pruned = torch.cat(grads_pruned)

            grads_alive_mean.append(grads_alive.mean().item())
            grads_alive_std.append(grads_alive.std().item())
            grads_pruned_mean.append(grads_pruned.mean().item())
            grads_pruned_std.append(grads_pruned.std().item())
            writer.add_scalar('avg_abs_grads/alive_mean', grads_alive.mean(), epoch_num)
            writer.add_scalar('avg_abs_grads/alive_var', grads_alive.var(), epoch_num)
            writer.add_scalar('avg_abs_grads/pruned_mean', grads_pruned.mean(), epoch_num)
            writer.add_scalar('avg_abs_grads/pruned_var', grads_pruned.var(), epoch_num)
    
    epoch_acc /= size
    epoch_loss /= size

    return epoch_acc, epoch_loss

def train(config, writer):
    device = config['device']
    model = load_model(config)
    train_loader, test_loader = load_dataset(config)
    print('Model has {} total params, including biases.'.format(model.get_total_params()))
    
    opt = get_opt(config, model)
    # scheduler = lr_scheduler.MultiStepLR(opt, milestones=[30,60], gamma=0.1)

    # Do SNIP if it is the case
    if config['prune_criterion'] == 'snip':
        keep_percentage = 1 - config['snip_sparsity']
        keep_masks = SNIP(model, keep_percentage, train_loader, device)
        apply_prune_mask(model, keep_masks)
    
    for epoch_num in range(1, config['epochs']+1):
        print('='*10 + ' Epoch ' + str(epoch_num) + ' ' + '='*10)

        model.train()
        # Anneal wdecay
        if config['anneal_lambda'] == True:
            opt.param_groups[0]['weight_decay'] = config['lambda']*(1-model.get_sparsity(config))

        train_acc, train_loss = epoch(epoch_num, train_loader, model, opt, writer, config)
        
        model.eval()
        # Comment this for now so that I can plot gradients
        # with torch.no_grad():
        test_acc, test_loss = epoch(epoch_num, test_loader, model, opt, writer, config)   
        
        if epoch_num%config['prune_freq'] == 0:
            if config['prune_criterion'] == 'magnitude':
                model.update_mask_magnitudes(config['prune_rate'])
            elif config['prune_criterion'] == 'flip':
                model.update_mask_flips(config['flip_threshold'])
            elif config['prune_criterion'] == 'random':
                model.update_mask_random(config['prune_rate'])
        
        print('Train - acc: {:>15.6f} loss: {:>15.6f}\nTest - acc: {:>16.6f} loss: {:>15.6f}'.format(
            train_acc, train_loss, test_acc, test_loss
        ))
        print('Sparsity : {:>15.4f}'.format(model.get_sparsity(config)))
        print('Wdecay : {:>15.6f}'.format(opt.param_groups[0]['weight_decay']))
        
        plot_stats(train_acc, train_loss, test_acc, test_loss, model, writer, epoch_num, config)
        # plot_weight_histograms(model, writer, epoch_num)

    plt.subplot(1, 2, 1)
    plt.errorbar(range(config['epochs']), grads_alive_mean, grads_alive_std, label='alive')
    plt.legend(); plt.grid()

    plt.subplot(1, 2, 2)
    plt.errorbar(range(config['epochs']), grads_pruned_mean, grads_pruned_std, label='pruned', color='orange')
    plt.legend(); plt.grid()
    
    plt.savefig('live_vs_pruned_fixed.png')


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['lenet300', 'lenet5', 'resnet18', 
                                                        'conv6', 'lenet300custom', 'lenet5custom',
                                                        'conv6custom'], default='lenet300')
    
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'], default='mnist')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    # Pruning
    parser.add_argument('--prune_criterion', type=str, choices=['magnitude', 'flip', 'random', 'snip'])
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
    parser.add_argument('--reg_type', type=str, choices=['wdecay', 'l1', 'l2'])
    parser.add_argument('--lambda', type=float, default=0)
    parser.add_argument('--anneal_lambda', dest='anneal_lambda', action='store_true')
    parser.add_argument('--no_anneal_lambda', dest='anneal_lambda', action='store_false')
    # Add noise or not
    parser.add_argument('--noise', dest='add_noise', action='store_true')
    parser.add_argument('--no_noise', dest='add_noise', action='store_false')
    # SNIP params
    parser.add_argument('--snip_sparsity', type=float, required=False, default=0.)
    config = vars(parser.parse_args())
    
    return config

if __name__ == "__main__":
    main()