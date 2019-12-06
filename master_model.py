import torch.nn as nn
import torch
import math
import numpy as np
from torch.distributions import Categorical

class MasterWrapper(object):
    def __init__(self, obj):
        self.obj = obj
        self.obj.total_params = self.obj.get_total_params()
        self.obj.save_weights()
        self.obj.save_rewind_weights()
        self.obj.instantiate_mask()
        self.obj.flip_counts = [torch.zeros_like(layer).to('cuda') for layer in self.parameters()]

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
        return sum([weights.numel() for weights in self.parameters()
                                if weights.requires_grad])
    
    def instantiate_mask(self):
        self.mask = [torch.ones_like(weights).to('cuda')
                        for weights in self.parameters()]
    
    def save_weights(self):
        self.saved_weights = [weights.clone().detach().to('cuda')
                                for weights in self.parameters()]
    
    def save_rewind_weights(self):
        self.rewind_weights = [weights.clone().detach().to('cuda')
                                for weights in self.parameters()]

    def rewind(self):
        for weights, rewind_weights, layer_mask in zip(self.parameters(), self.rewind_weights, self.mask):
            weights.data = rewind_weights.data*layer_mask
    
    def apply_mask(self):
        with torch.no_grad():
            for weights, layer_mask in zip(self.parameters(), self.mask):
                weights.grad.data = weights.grad.data*layer_mask
                weights.data = weights.data*layer_mask

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
        breakpoint()
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
            choice = torch.multinomial(torch.ones(valid_idxs.size(0)), to_prune_layer.item())
            selected_indices = a[valid_idxs[choice].squeeze().chunk(2)]
            layer_mask.data[selected_indices] = 0 
            layer.data = layer*layer_mask

    def get_sparsity(self):
        # Get the global sparsity rate
        with torch.no_grad():
            sparsity = 0
            for layer in self.parameters():
                sparsity += (layer==0).sum().item()
            return float(sparsity)/self.total_params

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
        flips_total = 0
        for layer_flips in self.flip_counts:
            flips_total += layer_flips.sum().item()
        return flips_total
    
    def get_total_flipped(self):
        # Get number of weights that flipped at least once
        total_flipped = 0
        for layer_flips in self.flip_counts:
            total_flipped += layer_flips[layer_flips >= 1].sum().item()
        return total_flipped
    
    def fuse_neurons(self, layer):
        # Fuze neurons of the network. Only do it on fc, without biases
        # for now
        layers = [layer for name,layer in self.named_parameters()
                    if 'weight' in name]
        # Fuse neurons in current layer and neurons in next layer
    
    def inject_noise(self, scaling_factor):
    # Inject Gaussian noise scaled by a factor into the gradients
        for layer in self.parameters():
            noise = torch.randn_like(layer)
            layer.grad += noise*scaling_factor
    
    def checkpoint(self, path):
        torch.save(self.state_dict(), path)