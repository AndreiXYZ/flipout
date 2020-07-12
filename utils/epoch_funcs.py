import math
import torch.nn as nn
import torch.nn.functional as F
import utils.utils as utils
import utils.getters as getters
from torch.nn.utils.clip_grad import clip_grad_norm_

def epoch_flips(epoch_num, loader, dataset_size, model, opt, writer, config):
    epoch_acc = 0
    epoch_loss = 0
    curr_lr = opt.param_groups[0]['lr']
    for batch_num, (x,y) in enumerate(loader):
        update_num = epoch_num*dataset_size/math.ceil(config['batch_size']) + batch_num
        opt.zero_grad()
        x = x.float().to(config['device'])
        y = y.to(config['device'])
        out = model.forward(x)

        sparsity = model.sparsity
        weight_penalty = getters.get_weight_penalty(model, config, epoch_num)

        if config['anneal_lambda'] == True:
            weight_penalty *= (1-sparsity)
        
        loss = F.cross_entropy(out, y) + weight_penalty
        
        if model.training:       
            model.save_weights()
            loss.backward()
            
            model.mask_grads(config)
            
            if config['add_noise']:
                if config['stop_noise_at']==-1 or epoch_num < config['stop_noise_at']:
                    noise_per_layer = model.inject_noise(config, epoch_num, curr_lr)
                    # for layer_noise, noisy_layer_name in zip(noise_per_layer, model.noisy_param_names):
                    #     writer.add_scalar('layerwise_noise/'+noisy_layer_name, layer_noise, update_num)

            if config['clip_grad']:
                total_norm = clip_grad_norm_(model.parameters(), config['max_norm'])
                # Log it every 100 updates or something
                if update_num % 100 == 0:
                    writer.add_scalar('grad_norm/total_norm', total_norm, update_num)
            
            opt.step()

            if config['opt'] == 'adam' or config['momentum']!=0.:
                model.mask_weights(config)
            
            # Monitor wegiths for flips
            flips_since_last = model.store_flips_since_last()

        epoch_acc += utils.accuracy(out, y)
        # multiply batch loss by batch size since the loss is averaged
        epoch_loss += x.size(0)*loss.item()

    epoch_acc /= dataset_size
    epoch_loss /= dataset_size

    return epoch_acc, epoch_loss

def regular_epoch(epoch_num, loader, dataset_size, model, opt, writer, config):
    epoch_acc = 0
    epoch_loss = 0
    curr_lr = opt.param_groups[0]['lr']

    for batch_num, (x,y) in enumerate(loader):
        update_num = epoch_num*dataset_size/math.ceil(config['batch_size']) + batch_num
        opt.zero_grad()
        x = x.float().to(config['device'])
        y = y.to(config['device'])
        out = model.forward(x)

        sparsity = model.sparsity
        weight_penalty = getters.get_weight_penalty(model, config, epoch_num)

        if config['anneal_lambda'] == True:
            weight_penalty *= (1-sparsity)

        loss = F.cross_entropy(out, y) + weight_penalty
        
        if model.training:       
            loss.backward()
            model.mask_grads(config)
            
            if config['add_noise']:
                if config['stop_noise_at']==-1 or epoch_num < config['stop_noise_at']:
                    noise_per_layer = model.inject_noise(config, epoch_num, curr_lr)

            if config['clip_grad']:
                total_norm = clip_grad_norm_(model.parameters(), config['max_norm'])
                # Log it every 100 updates or something
                if update_num % 100 == 0:
                    writer.add_scalar('grad_norm/total_norm', total_norm, update_num)
            
            opt.step()
        
            if config['opt'] == 'adam' or config['momentum']!=0.:
                model.mask_weights(config)
            
        epoch_acc += utils.accuracy(out, y)
        # multiply batch loss by batch size since the loss is averaged
        epoch_loss += x.size(0)*loss.item()
    

    epoch_acc /= dataset_size
    epoch_loss /= dataset_size

    return epoch_acc, epoch_loss