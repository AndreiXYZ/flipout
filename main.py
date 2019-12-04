import torch, argparse
import numpy as np
import math
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from utils import *
from models import *
from data_loaders import *
from master_model import MasterWrapper

def epoch(epoch_num, loader, size, model, opt, criterion, writer, config):
    epoch_acc = 0
    epoch_loss = 0
    temperature = (config['lr']/2)*(10**-12)
    scaling_factor = math.sqrt(2*config['lr']*temperature)*(1/config['lr'])

    for batch_num, (x,y) in enumerate(loader):
        
        update_num = epoch_num*size/math.ceil(config['batch_size']) + batch_num
        opt.zero_grad()
        x = x.float().to(config['device'])
        y = y.to(config['device'])
        out = model.forward(x).squeeze()
        loss = criterion(out, y)
        if model.training:
            model.save_weights()
            loss.backward()
            model.inject_noise(scaling_factor)
            model.apply_mask()
            opt.step()
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

    if config['model'] == 'lenet300':
        model = LeNet_300_100()
    elif config['model'] == 'lenet5':
        model = LeNet5()

    
    model = MasterWrapper(model).to(device)
    print('Model has {} total params, including biases.'.format(model.get_total_params()))

    if config['dataset']=='mnist':
        train_loader, train_size, test_loader, test_size = get_mnist_loaders(config)
    elif config['dataset'] == 'cifar10':
        train_loader, train_size, test_loader, test_size = get_cifar10_loaders(config)

    opt = optim.RMSprop(model.parameters(), lr=config['lr'], weight_decay=1e-5)

    # scheduler = lr_scheduler.OneCycleLR(opt,
                                        
    #                                 )

    criterion = nn.CrossEntropyLoss()

    for epoch_num in range(config['epochs']):
        print('='*10 + ' Epoch ' + str(epoch_num) + ' ' + '='*10)

        model.train()
        train_acc, train_loss = epoch(epoch_num, train_loader, train_size, model, opt, criterion, writer, config)

        model.eval()
        with torch.no_grad():
            test_acc, test_loss = epoch(epoch_num, test_loader, test_size, model, opt, criterion, writer, config)   

        print('Train - acc: {:>15.8f} loss: {:>15.8f}\nTest - acc: {:>16.8f} loss: {:>15.8f}'.format(
            train_acc, train_loss, test_acc, test_loss
        ))

        if config['rewind_to'] > 1 and (epoch_num+1)==config['rewind_to']:
            model.save_rewind_weights()
        
        if (epoch_num+1)%config['prune_freq'] == 0:

            if config['prune_criterion'] == 'magnitude':
                model.update_mask_magnitudes(config['prune_rate'])
            elif config['prune_criterion'] == 'flip':
                model.update_mask_flips(config['flip_prune_threshold'])

            if config['rewind_to'] > 1:
                model.rewind()
        
                
        writer.add_scalar('acc/train', train_acc, epoch_num)
        writer.add_scalar('acc/test', test_acc, epoch_num)
        writer.add_scalar('loss/train', train_loss, epoch_num)
        writer.add_scalar('loss/test', test_loss, epoch_num)
        writer.add_scalar('sparsity', model.get_sparsity(), epoch_num)
        # Visualise histogram of flips
        writer.add_histogram('layer 0 flips hist.', model.flip_counts[0].flatten(), epoch_num)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['lenet300', 'lenet5'], default='lenet300')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'], default='mnist')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_every', type=int, default=-1)
    # Pruning
    parser.add_argument('--prune_criterion', type=str, choices=['magnitude', 'flip'])
    parser.add_argument('--prune_freq', type=int, default=2)
    parser.add_argument('--prune_rate', type=float, default=0.2) # for magnitude pruning
    parser.add_argument('--flip_prune_threshold', type=int, default=1) # for flip pruning
    parser.add_argument('--rewind_to', type=int, default=-1) # for rewinding the weights
    # Run comment
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to add to tensorboard text ')
    # Stochastic noise temperature parameter
    # parser.add_argument('--temperature', type=float, default=0,
    #                     help='Temperature parameter used for injecting noise to the gradient.')
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