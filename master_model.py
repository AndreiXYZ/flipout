import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np

from torch.distributions import Categorical
from utils import *

class MasterWrapper(object):
    def __init__(self, obj):
        self.obj = obj
        self.obj.total_params = self.obj.get_total_params()
        self.obj.save_weights()
        self.obj.instantiate_mask()
        self.obj.flip_counts = [torch.zeros_like(layer, dtype=torch.float).to('cuda') for layer in self.parameters()]
        self.obj.ema_flip_counts = [torch.zeros_like(layer, dtype=torch.float).to('cuda') for layer in self.parameters()]
        self.live_connections = None
        self.sparsity = 0

    def __getattr__(self, name):
        # Override getattr such that it calls the wrapped object's attrs
        func = getattr(self.__dict__['obj'], name)
        if callable(func):
            def wrapper(*args, **kwargs):
                ret = func(*args, **kwargs)
                return ret
            return wrapper
        else:
            return func


class MasterModel(nn.Module):
    def __init__(self):
        super(MasterModel, self).__init__()
    
    def get_total_params(self):
        with torch.no_grad():
            return sum([weights.numel() for weights in self.parameters()
                                if weights.requires_grad])

    def get_sparsity(self, config):
    #Get the global sparsity rate
        with torch.no_grad():
            sparsity = 0
            if 'custom' in config['model']:
                for layer in self.parameters():
                    relu_weights = F.relu(layer)
                    sparsity += (layer<=0).sum().item()
            else:
                for layer in self.parameters():
                    sparsity += (layer==0).sum().item()

        return float(sparsity)/self.get_total_params()

    def instantiate_mask(self):
        self.mask = [torch.ones_like(layer, dtype=torch.bool).to('cuda') for layer in self.parameters()]

    def save_weights(self):
        self.saved_weights = [layer.data.detach().clone().to('cuda') for layer in self.parameters()]
    
    def save_grads(self):
        self.saved_grads = [layer.grad.clone().to('cuda') for layer in self.parameters()]

    def save_rewind_weights(self):
        self.rewind_weights = [layer.data.detach().clone().to('cuda') for layer in self.parameters()]

    def rewind(self):
        for weights, rewind_weights, layer_mask in zip(self.parameters(), self.rewind_weights, self.mask):
            weights.data = rewind_weights.data*layer_mask
    
    def mask_weights(self, config):
        with torch.no_grad():
            if 'custom' in config['model']:
                for weights in self.parameters():
                    layer_mask = F.relu(weights)>0
                    weights.data = weights.data*layer_mask
            else:
                for weights, layer_mask in zip(self.parameters(), self.mask):
                    weights.data = weights.data*layer_mask
    
    
    def mask_grads(self, config):
        with torch.no_grad():
            if 'custom' in config['model']:
                for weights in self.parameters():
                    layer_mask = F.relu(weights)>0
                    weights.grad.data = weights.grad.data*layer_mask
            else:
                for weights, layer_mask in zip(self.parameters(), self.mask):
                    weights.grad.data = weights.grad.data*layer_mask

    
    def update_mask_magnitudes(self, rate):
        # Prune parameters of the network according to lowest magnitude
        with torch.no_grad():
            print('OK')
            for layer, layer_mask in zip(self.parameters(), self.mask):
                num_pruned = (layer_mask==0).sum().item()
                num_unpruned = layer.numel() - num_pruned
                num_to_prune = num_pruned + int(rate*num_unpruned)
                
                to_prune = layer.view(-1).abs().argsort(descending=False)[:num_to_prune]
                layer_mask.view(-1)[to_prune] = 0
                layer.data = layer*layer_mask


    def update_mask_flips(self, threshold):
        with torch.no_grad():
        # Prune parameters based on sign flips
            for layer, layer_flips, layer_mask in zip(self.parameters(), self.flip_counts, self.mask):
                # Get parameters whose flips are above a threshold and invert for masking
                flip_mask = ~(layer_flips >= threshold)
                layer_mask.data = flip_mask*layer_mask
                layer_flips.data *= layer_mask
                layer.data = layer*layer_mask
    

    def update_mask_topflips(self, rate, use_ema):
        with torch.no_grad():
            # Prune parameters of the network according to highest flips
            # Flatten everything
            if use_ema:
                flip_cts = self.ema_flip_counts
            else:
                flip_cts = self.flip_counts
            
            flip_cts = torch.cat([layer_flips.clone().view(-1) 
                                    for layer_flips in flip_cts])
            flat_mask = torch.cat([layer_mask.clone().view(-1)
                                    for layer_mask in self.mask])
            flat_params = torch.cat([layer.clone().view(-1)
                                    for layer in self.parameters()])
            
            # Get first n% flips
            num_pruned = (flat_mask==0).sum().item()
            num_to_prune = int((self.total_params - num_pruned)*rate)
            to_prune = flip_cts.argsort(descending=True)[:num_to_prune]
            # Grab only those who are different than 0
            to_prune = to_prune[flip_cts[to_prune] > 0]
            # Update mask and flip counts
            flat_mask[to_prune] = 0
            flip_cts[to_prune] = 0
            # Replace mask and update parameters
            self.mask = self.unflatten_tensor(flat_mask, self.mask)
            
            for layer, layer_mask in zip(self.parameters(), self.mask):
                layer.data = layer*layer_mask
            

    def update_mask_topflips_layerwise(self, rate):
        with torch.no_grad():
            for layer, layer_mask, layer_flips in zip(self.parameters(), self.mask, self.flip_counts):
                # Calc how many weights we still need to prune
                num_pruned = (layer_mask==0).sum().item()
                num_to_prune = (layer.numel() - num_pruned)*rate
                
                # Get index of the weights that are to be pruned
                to_prune = layer_flips.view(-1).argsort(descending=True)[:int(num_to_prune)]
                # Only prune those that have more than 0 flips
                to_prune = to_prune[layer_flips.view(-1)[to_prune] > 0]
                # Update mask and multiply layer by mask
                layer_mask.data.view(-1)[to_prune] = 0
                layer_flips.data.view(-1)[to_prune] = 0
                
                layer.data = layer*layer_mask
    
    def update_mask_random(self, rate, config):
        # Get prob distribution
        with torch.no_grad():
            flat_mask = torch.cat([layer_mask.view(-1) for layer_mask in self.mask])
            num_unpruned = (1-self.get_sparsity(config))*self.total_params
            num_to_prune = num_unpruned*rate
            # Get only the values which are nonzero
            idx_nonzeros = (flat_mask!=0.).nonzero().view(-1)
            
            # Select randomly from those
            idxs = np.random.choice(idx_nonzeros.cpu().numpy(), int(num_to_prune), replace=False)
            flat_mask[idxs] = 0

            self.mask = self.unflatten_tensor(flat_mask, self.mask)

            for layer, layer_mask in zip(self.parameters(), self.mask):
                layer.data = layer*layer_mask
            

    def store_flips_since_last(self):
    # Retrieves how many params have flipped compared to previously saved weights
        with torch.no_grad():
            num_flips = 0
            for curr_weights, prev_weights, layer_flips, layer_mask in zip(self.parameters(), self.saved_weights, self.flip_counts,
                                                                self.mask):
                curr_signs = curr_weights.sign()
                prev_signs = prev_weights.sign()
                flipped = ~curr_signs.eq(prev_signs)
                layer_flips += flipped
                layer_flips *= layer_mask
                num_flips += flipped.sum()
        return num_flips


    def store_ema_flip_counts(self, beta):
        with torch.no_grad():
            for layer_flips, layer_ema_flips, layer_mask in zip(self.flip_counts, self.ema_flip_counts, self.mask):
                layer_ema_flips.data = beta*layer_ema_flips + (1-beta)*layer_flips
                layer_ema_flips.data = layer_ema_flips*layer_mask


    def get_flips_total(self):
        # Get total number of flips
        with torch.no_grad():
            flips_total = 0
            for layer_flips in self.flip_counts:
                flips_total += layer_flips.sum().item()
        return flips_total
    
    def get_total_flipped(self):
        # Get number of weights that flipped at least once
        with torch.no_grad():
            total_flipped = 0
            for layer_flips in self.flip_counts:
                total_flipped += layer_flips[layer_flips >= 1].sum().item()
        return total_flipped

    def inject_noise(self, config, epoch_num, curr_lr):
    # Inject Gaussian noise scaled by a factor into the gradients
        with torch.no_grad():
            noise_per_layer = []
            if 'custom' not in config['model']:
                for layer, layer_mask in zip(self.parameters(),self.mask):
                    # Add noise equal to layer-wise l2 norm of params
                    noise = torch.randn_like(layer)
                    scaling_factor = layer.grad.norm(p=2)/math.sqrt(layer.numel())
                    # Scale noise by LR
                    if config['scale_noise_by_lr']:
                        scaling_factor *= config['lr']/curr_lr

                    layer.grad.data += noise*scaling_factor
                    # Append to list for logging purposes
                    noise_per_layer.append(scaling_factor)
                    # Finally, mask gradient for pruned weights
                    layer.grad.data *= layer_mask
        
            else:
                for layer in self.parameters():
                    layer_mask = F.relu(layer)>0
                    noise = torch.randn_like(layer)
                    scaling_factor = layer.grad.norm(p=1)/layer.numel()
                    noise_per_layer.append(scaling_factor)
                    # Only add noise to elements which are nonzero
                    layer.grad.data += noise*scaling_factor
                    layer.grad.data *= layer_mask

        return noise_per_layer

    def get_sign_percentages(self):
        # Get total num of weights
        total_remaining_weights = 0
        remaining_pos = 0
        with torch.no_grad():
            for layer in self.parameters():
                flat_layer = layer.flatten()
                remaining = flat_layer[flat_layer != 0]

                total_remaining_weights += remaining.numel()
                remaining_pos += remaining[remaining > 0].numel()
    
        return total_remaining_weights, remaining_pos


    @staticmethod
    def unflatten_tensor(flat_tensor, tensor_list):
        # Unflatten a tensor according to a list of tensors such that it matches
        idx = 0
        result_list = []
        for tensor in tensor_list:
            tensor_numel = tensor.numel()
            result_list.append(flat_tensor[idx:idx+tensor_numel].view_as(tensor))
            idx += tensor_numel
        return result_list