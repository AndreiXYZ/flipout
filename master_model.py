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
        self.obj.flip_counts = [torch.zeros_like(layer, dtype=torch.short).to('cuda') for layer in self.parameters()]
        
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
    
    def get_sparsity(self,config):
    # Get the global sparsity rate
        with torch.no_grad():
            sparsity = 0
            if 'custom' in config['model']:
                for layer in self.parameters():
                    relu_weights = F.relu(layer)
                    sparsity += (layer<=0).sum().item()
            else:
                for layer in self.parameters():
                    sparsity += (layer==0).sum().item()
    
        return float(sparsity)/self.total_params

    def get_flattened_params(self):
        return torch.cat([layer.view(-1) for layer in self.parameters()])

    def get_weight_penalty(self, config):
        penalty = None
        if config['reg_type'] == 'l1':
            for layer in self.parameters():
                if penalty is None:
                    penalty = layer.norm(p=1)
                else:
                    penalty = penalty + layer.norm(p=1)
        
        elif config['reg_type'] == 'l2':
            for layer in self.parameters():
                if penalty is None:
                    penalty = layer.norm(p=2)**2
                else:
                    penalty = penalty + layer.norm(p=2)**2

        else:
            penalty = 0

        return penalty

    def instantiate_mask(self):
        self.mask = [torch.ones_like(layer, dtype=torch.bool).to('cuda') for layer in self.parameters()]
    
    def save_weights(self):
        self.saved_weights = [layer.data.detach().clone().to('cuda')
                                for layer in self.parameters()]
    
    def save_grads(self):
        self.saved_grads = [layer.grad.clone().to('cuda')
                            for layer in self.parameters()]
    
    def save_rewind_weights(self):
        self.rewind_weights = [weights.detach().clone().to('cuda')
                                for weights in self.parameters()]

    def rewind(self):
        for weights, rewind_weights, layer_mask in zip(self.parameters(), self.rewind_weights, self.mask):
            weights.data = rewind_weights.data*layer_mask
    
    def apply_mask(self, config):
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
            for layer, layer_mask in zip(self.parameters(), self.mask):
                flat_layer = layer.view(-1)
                indices = flat_layer.abs().argsort(descending=False)

                num_pruned = (layer_mask==0).sum().item()
                num_unpruned = layer_mask.numel() - num_pruned
                to_prune = num_pruned + int(rate*num_unpruned)
                
                indices = indices[:to_prune]
                mask = layer_mask.view(-1).clone()
                mask[indices] = 0
                layer_mask.data = mask.view_as(layer_mask)
                layer.data = layer*layer_mask

    def update_mask_magnitude_global(self, rate):
        with torch.no_grad():
            num_els = self.get_total_params()
            curr_sparsity = self.get_sparsity()
            num_unpruned = 1 - num_els*curr_sparsity
            num_to_prune = rate*num_unpruned
            # TODO


    def update_mask_flips(self, threshold):
        with torch.no_grad():
        # Prune parameters based on sign flips
            for layer, layer_flips, layer_mask in zip(self.parameters(), self.flip_counts, self.mask):
                # Get parameters whose flips are above a threshold and invert for masking
                flip_mask = ~(layer_flips >= threshold)
                layer_mask.data = flip_mask*layer_mask
                layer.data = layer*layer_mask

    def update_mask_random(self, rate):
        # Get prob distribution
        distribution = torch.Tensor([layer.numel() for layer in self.parameters()
                                    if layer.requires_grad])
        distribution /= distribution.sum()

        to_prune = (1-self.get_sparsity())*rate
        to_prune_absolute = math.ceil(self.get_total_params()*to_prune)
        
        # Get how many params to remove per layer
        distribution *= to_prune_absolute
        distribution = distribution.int()

        # Sample to_prune times from the nonzero elemnets
        for layer, layer_mask, to_prune_layer in zip(self.parameters(), self.mask, distribution):
            valid_idxs = layer_mask.data.nonzero()
            choice = torch.randperm(valid_idxs.size(0))[:to_prune_layer]
            selected_indices = valid_idxs[choice].chunk(2,dim=1)
            layer_mask.data[selected_indices] = 0 
            layer.data = layer*layer_mask

    def update_mask_gradpruned(self, threshold):
        with torch.no_grad():
            # First, check if there are any pruned weights
            for layer, layer_mask in zip(self.parameters(), self.mask):
                if layer.grad[layer_mask==0].numel() > 0:
                    layer_mask[layer.abs() <= threshold] = 0
                    layer.data = layer*layer_mask

    def store_flips_since_last(self):
    # Retrieves how many params have flipped compared to previously saved weights
        with torch.no_grad():
            num_flips = 0
            
            for curr_weights, prev_weights, layer_flips in zip(self.parameters(), self.saved_weights, self.flip_counts):
                curr_signs = curr_weights.sign()
                prev_signs = prev_weights.sign()
                flipped = ~curr_signs.eq(prev_signs)
                layer_flips += flipped
                num_flips += flipped.sum()
        return num_flips

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

    def inject_noise(self, config):
    # Inject Gaussian noise scaled by a factor into the gradients
        with torch.no_grad():
            noise_per_layer = []
            if 'custom' not in config['model']:
                for layer, layer_mask in zip(self.parameters(),self.mask):
                    # Add noise equal to layer-wise l2 norm of params
                    noise = torch.randn_like(layer)
                    scaling_factor = layer.grad.norm(p=2)/math.sqrt(layer.numel())
                    layer.grad.data += noise*scaling_factor
                    # Append to list for logging purposes
                    noise_per_layer.append(scaling_factor + grad_pruned)
                    # Finally, mask gradient for pruned weights
                    layer.grad.data *= layer_mask
        
            else:
                for layer in self.parameters():
                    layer_mask = F.relu(layer)>0
                    noise = torch.randn_like(layer)
                    scaling_factor = layer.grad.norm(p=2)/math.sqrt(layer.numel())
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