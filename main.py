import torch, argparse, sys
import numpy as np
import math
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from utils import set_seed, get_opt, construct_run_name
from models import *
from data_loaders import *
from master_model import MasterWrapper

def epoch(epoch_num, loader,  model, opt, criterion, writer, config):
    epoch_acc = 0
    epoch_loss = 0
    size = len(loader.dataset)
    
    for batch_num, (x,y) in enumerate(loader):
        update_num = epoch_num*size/math.ceil(config['batch_size']) + batch_num
        opt.zero_grad()
        x = x.float().to(config['device'])
        y = y.to(config['device'])
        out = model.forward(x)
        loss = criterion(out, y)

        if model.training:
            writer.add_scalar('sparsity/sparsity_before_step', model.get_sparsity(), update_num)
            model.save_weights()
            loss.backward()
            
            model.apply_mask()
            model.inject_noise()
            opt.step()
            
            writer.add_scalar('sparsity/sparsity_after_step', model.get_sparsity(), update_num)
            # Monitor wegiths for flips
            flips_since_last = model.store_flips_since_last()
            flips_total = model.get_flips_total()
            flipped_total = model.get_total_flipped()
            writer.add_scalar('flips/flips_since_last', flips_since_last, update_num)
            writer.add_scalar('flips/percentage_since_last', float(flips_since_last)/model.total_params, update_num)
            writer.add_scalar('flips/flipped_total', flipped_total, update_num)

        preds = out.argmax(dim=1, keepdim=True).squeeze()
        correct = preds.eq(y).sum().item()
        
        epoch_acc += correct
        epoch_loss += loss.item()
    
    epoch_acc /= size
    epoch_loss /= size

    return epoch_acc, epoch_loss 

def train(config, writer):
    device = config['device']
    model = load_model(config)
    train_loader, test_loader = load_dataset(config)
    print('Model has {} total params, including biases.'.format(model.get_total_params()))
    
    opt = get_opt(config, model.parameters())

    criterion = nn.CrossEntropyLoss()

    for epoch_num in range(config['epochs']):
        print('='*10 + ' Epoch ' + str(epoch_num) + ' ' + '='*10)

        model.train()
        train_acc, train_loss = epoch(epoch_num, train_loader, model, opt, criterion, writer, config)
        
        model.eval()
        with torch.no_grad():
            test_acc, test_loss = epoch(epoch_num, test_loader, model, opt, criterion, writer, config)   
        
        print('Train - acc: {:>15.6f} loss: {:>15.6f}\nTest - acc: {:>16.6f} loss: {:>15.6f}'.format(
            train_acc, train_loss, test_acc, test_loss
        ))
        print('Sparsity : {:>10.4f}'.format(model.get_sparsity()))
        
        if (epoch_num+1)%config['prune_freq'] == 0:
            if config['prune_criterion'] == 'magnitude':
                model.update_mask_magnitudes(config['prune_rate'])
            elif config['prune_criterion'] == 'flip':
                model.update_mask_flips(config['flip_prune_threshold'])
            elif config['prune_criterion'] == 'random':
                model.update_mask_random(config['prune_rate'])
        
        
        writer.add_scalar('acc/train', train_acc, epoch_num)
        writer.add_scalar('acc/test', test_acc, epoch_num)
        writer.add_scalar('loss/train', train_loss, epoch_num)
        writer.add_scalar('loss/test', test_loss, epoch_num)
        writer.add_scalar('sparsity/sparsity', model.get_sparsity(), epoch_num)
        writer.add_scalar('lr', opt.param_groups[0]['lr'], epoch_num)
        # Visualise histogram of flips
        writer.add_histogram('layer 0 flips hist.', model.flip_counts[0].flatten(), epoch_num)

        
        for name,layer in model.named_parameters():
            if layer.requires_grad:
                layer_histogram = layer.clone().detach().flatten()
                # Get only nonzeros for visibility
                layer = layer[layer!=0]
                if 'weight' in name:
                    writer.add_histogram('weights/'+name, layer.clone().detach().flatten(), epoch_num)
                elif 'bias' in name:
                    writer.add_histogram('biases/'+name, layer.clone().detach().flatten(), epoch_num)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['lenet300', 'lenet5', 'resnet18', 'vgg11'], default='lenet300')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'], default='mnist')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    # Pruning
    parser.add_argument('--prune_criterion', type=str, choices=['magnitude', 'flip', 'random'])
    parser.add_argument('--prune_freq', type=int, default=2)
    parser.add_argument('--prune_rate', type=float, default=0.2) # for magnitude pruning
    parser.add_argument('--flip_prune_threshold', type=int, default=1) # for flip pruning
    # Run comment
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to add to tensorboard text ')
    # Optimizer args
    parser.add_argument('--opt', type=str, choices=['sgd', 'rmsprop', 'adam'])
    parser.add_argument('--wdecay', type=float, default=0)

    config = vars(parser.parse_args())
    
    
    # Ensure experiment is reproducible.
    # Results may vary across machines!
    set_seed(config['seed'])

    run_hparams = construct_run_name(config)
    writer = SummaryWriter(comment=run_hparams)
    writer.add_text('config', run_hparams)
    writer.add_text('comment', config['comment'])
    train(config, writer)

if __name__ == "__main__":
    main()