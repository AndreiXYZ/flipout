import math
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy, get_weight_penalty

def epoch_flips(epoch_num, loader, size, model, opt, writer, config):
    epoch_acc = 0
    epoch_loss = 0
    for batch_num, (x,y) in enumerate(loader):
        update_num = epoch_num*size/math.ceil(config['batch_size']) + batch_num
        opt.zero_grad()
        x = x.float().to(config['device'])
        y = y.to(config['device'])
        out = model.forward(x)

        sparsity = model.sparsity
        weight_penalty = get_weight_penalty(model, config)

        if config['anneal_lambda'] == True:
            weight_penalty *= (1-sparsity)
        
        loss = F.cross_entropy(out, y) + weight_penalty*config['lambda']
        
        if model.training:       
            model.save_weights()
            loss.backward()

            model.mask_grads(config)
            
            if config['add_noise']:
                noise_per_layer = model.inject_noise(config)

            opt.step()

            if config['opt'] == 'adam':
                model.mask_weights(config)
            
            # Monitor wegiths for flips
            flips_since_last = model.store_flips_since_last()

        epoch_acc += accuracy(out, y)
        epoch_loss += loss.item()


    epoch_acc /= size
    epoch_loss /= size

    return epoch_acc, epoch_loss


def epoch_l0(epoch_num, loader, size, model, opt, writer, config):
    epoch_acc = 0
    epoch_loss = 0

    # If it's not training, load the EMA parameters
    if not model.training:
        if model.beta_ema > 0:
            old_params = model.get_params()
            model.load_ema_params()
    
    for batch_num, (x,y) in enumerate(loader):
        update_num = epoch_num*size/math.ceil(config['batch_size']) + batch_num
        opt.zero_grad()
        x = x.float().to(config['device'])
        y = y.to(config['device'])
        out = model.forward(x)
        
        loss = F.cross_entropy(out, y) + model.regularization()
        
        if model.training:       
            loss.backward()
            opt.step()
            # clamp the parameters
            layers = model.layers #if not args.multi_gpu else model.module.layers
            for k, layer in enumerate(layers):
                layer.constrain_parameters()
            # Update the EMA
            if model.beta_ema > 0.:
                model.update_ema()

        epoch_acc += accuracy(out, y)
        epoch_loss += loss.item()

    if not model.training:
        if model.beta_ema > 0:
            model.load_params(old_params)
    
    epoch_acc /= size
    epoch_loss /= size

    return epoch_acc, epoch_loss


def regular_epoch(epoch_num, loader, size, model, opt, writer, config):
    epoch_acc = 0
    epoch_loss = 0
    for batch_num, (x,y) in enumerate(loader):
        update_num = epoch_num*size/math.ceil(config['batch_size']) + batch_num
        opt.zero_grad()
        x = x.float().to(config['device'])
        y = y.to(config['device'])
        out = model.forward(x)

        sparsity = model.sparsity
        weight_penalty = get_weight_penalty(model, config)

        if config['anneal_lambda'] == True:
            weight_penalty *= (1-sparsity)
        
        loss = F.cross_entropy(out, y) + weight_penalty*config['lambda']
        
        if model.training:       
            loss.backward()
            model.mask_grads(config)
            
            if config['add_noise']:
                noise_per_layer = model.inject_noise(config)

            opt.step()

            if config['opt'] == 'adam':
                model.mask_weights(config)
            
        epoch_acc += accuracy(out, y)
        epoch_loss += loss.item()


    epoch_acc /= size
    epoch_loss /= size

    return epoch_acc, epoch_loss


def get_epoch_type(config):
    if config['prune_criterion'] == 'flip' or config['prune_criterion'] == 'topflip':
        return epoch_flips
    elif config['prune_criterion'] == 'l0':
        return epoch_l0
    
    return regular_epoch