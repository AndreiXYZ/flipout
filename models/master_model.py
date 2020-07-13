import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np
import sys
from torch.distributions import Categorical
from utils import *

class CustomDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            # Return attr of data parallel
            return super().__getattr__(name)
        except AttributeError:
            # If it doesn't exist return attr of wrapped model
            return getattr(self.module, name)
    
def init_attrs(model, config):
    from itertools import chain
    # Only prune linear and conv2d models (not batchnorm)
    prunable_modules = [nn.Linear, nn.Conv2d]

    if config['prune_bnorm']:
        prunable_modules.append(nn.BatchNorm2d)

    prunable_modules = tuple(prunable_modules)
    
    if config['prune_bias']:
        model.prunable_params = list(chain.from_iterable(
        [[layer.weight, layer.bias] for layer in model.modules()
                                 if isinstance(layer, prunable_modules)
                                 and layer.weight.requires_grad
                                 and layer.bias.requires_grad]
        ))
    else:
        model.prunable_params = [module.weight for module in model.modules()
                                 if isinstance(module, prunable_modules)
                                 and module.weight.requires_grad
                                 ]
    
    # Generate a list of names of modules we want to inject noise into
    model.noisy_params = model.prunable_params if config['noise_only_prunable'] \
                        else [layer for layer in model.parameters()]
    
    model.noisy_param_names = []

    for noisy_param in model.noisy_params:
        for name,param in model.named_parameters():
            if param is noisy_param:
                model.noisy_param_names.append(name)

    model.total_prunable = sum([layer.numel() for layer in model.prunable_params])
    print('Total prunable params of model:', model.total_prunable)
    model.save_weights()
    model.instantiate_mask()

    if config['prune_criterion'] == 'flipout':
        model.flip_counts = [torch.zeros_like(layer, dtype=torch.float).to('cuda') 
                            for layer in model.prunable_params]
    
    model.live_connections = None
    model.sparsity = 0

class MasterModel(nn.Module):
    def __init__(self):
        super(MasterModel, self).__init__()
    
    def get_total_params(self):
        with torch.no_grad():
            
            num_weights = sum([module.weight.numel() for module in self.modules()
                        if hasattr(module, 'weight')])
            
            num_bias = sum([module.bias.numel() for module in self.modules()
                            if hasattr(module, 'bias') and module.bias is not None])
        
        return num_weights, num_bias

    def get_sparsity(self, config):
    #Get the global sparsity rate
        with torch.no_grad():
            sparsity = 0
            for layer in self.prunable_params:
                sparsity += (layer==0).sum().item()

        return float(sparsity)/self.total_prunable

    def instantiate_mask(self):
        self.mask = [torch.ones_like(layer, dtype=torch.bool).to('cuda') 
                     for layer in self.prunable_params]

    def save_weights(self):
        self.saved_weights = [layer.data.detach().clone().to('cuda') 
                              for layer in self.prunable_params]
    
    def save_grads(self):
        self.saved_grads = [layer.grad.clone().to('cuda') 
                            for layer in self.parameters()]

    def mask_weights(self, config):
        with torch.no_grad():
            for weights, layer_mask in zip(self.prunable_params, self.mask):
                weights.data = weights.data*layer_mask
    
    
    def mask_grads(self, config):
        with torch.no_grad():
            for weights, layer_mask in zip(self.prunable_params, self.mask):
                weights.grad.data = weights.grad.data*layer_mask
    
    def update_mask_magnitudes(self, rate):
        # Prune parameters of the network according to lowest magnitude
        with torch.no_grad():
            for layer, layer_mask in zip(self.prunable_params, self.mask):
                num_pruned = (layer_mask==0).sum().item()
                num_unpruned = layer.numel() - num_pruned
                num_to_prune = num_pruned + int(rate*num_unpruned)
                
                to_prune = layer.view(-1).abs().argsort(descending=False)[:num_to_prune]
                layer_mask.view(-1)[to_prune] = 0
                layer.data = layer*layer_mask
    
    def update_mask_threshold(self, threshold):
        # Prune all weights below a threshold
        with torch.no_grad():
            for layer, layer_mask in zip(self.prunable_params, self.mask):
                layer_mask.data = ~(layer.abs() < threshold)*layer_mask
                
                num_nonzeros = (layer_mask.view(-1)==0).sum()
                layer.data = layer*layer_mask
    
    def update_mask_global_magnitudes(self, rate):
        with torch.no_grad():
            flat_params = torch.cat([layer.view(-1) for layer in self.prunable_params])
            flat_mask = torch.cat([layer_mask.view(-1) for layer_mask in self.mask])

            num_pruned = (flat_params==0).sum().item()
            num_to_prune = int((self.total_prunable - num_pruned)*rate)

            to_prune = flat_params.abs().argsort(descending=False)[:num_pruned+num_to_prune]
            # This should be all zeros
            flat_mask[to_prune] = 0.
            self.mask = self.unflatten_tensor(flat_mask, self.mask)
            # Now update the weights
            for layer, layer_mask in zip(self.prunable_params, self.mask):
                layer.data = layer * layer_mask
    
    def update_mask_flipout(self, rate, flipout_p):
        with torch.no_grad():
            flat_mask = torch.cat([layer_mask.view(-1) for layer_mask in self.mask])
            flat_magnitudes = torch.cat([layer.view(-1) for layer in self.prunable_params])
            flip_cts = torch.cat([layer_flips.view(-1) for layer_flips in self.flip_counts])

            # Determine how many we need to prune
            num_pruned = (flat_mask==0).sum().item()
            num_to_prune = int((self.total_prunable-num_pruned)*rate)

            # Do weight divided by number of flips
            # add a 1 to denominator to avoid division by 0
            criterion = flat_magnitudes.pow(flipout_p)/(flip_cts+1)
            to_prune = criterion.argsort(descending=False)[:num_pruned+num_to_prune]
            flat_mask[to_prune] = 0.
            self.mask = self.unflatten_tensor(flat_mask, self.mask)

            for layer, layer_mask in zip(self.prunable_params, self.mask):
                layer.data = layer*layer_mask
    
    def update_mask_random(self, rate, config):
        # Get prob distribution
        with torch.no_grad():
            flat_mask = torch.cat([layer_mask.view(-1) for layer_mask in self.mask])
            num_unpruned = (1-self.get_sparsity(config))*self.total_prunable
            num_to_prune = num_unpruned*rate
            # Get only the values which are nonzero
            idx_nonzeros = (flat_mask!=0.).nonzero().view(-1)
            
            # Select randomly from those
            idxs = np.random.choice(idx_nonzeros.cpu().numpy(), int(num_to_prune), replace=False)
            flat_mask[idxs] = 0

            self.mask = self.unflatten_tensor(flat_mask, self.mask)

            for layer, layer_mask in zip(self.prunable_params, self.mask):
                layer.data = layer*layer_mask
    
    def store_flips_since_last(self):
    # Retrieves how many params have flipped compared to previously saved weights
        with torch.no_grad():
            num_flips = 0
            for curr_weights, prev_weights, layer_flips, layer_mask in zip(self.prunable_params, self.saved_weights, self.flip_counts,
                                                                self.mask):
                curr_signs = curr_weights.sign()
                prev_signs = prev_weights.sign()
                flipped = ~curr_signs.eq(prev_signs)
                layer_flips += flipped
                layer_flips *= layer_mask
                num_flips += flipped.sum()
        return num_flips

    def inject_noise(self, config, epoch_num, curr_lr):
    # Inject Gaussian noise scaled by a factor into the gradients
        with torch.no_grad():
            noise_per_layer = []

            for layer in self.noisy_params:
                # Add noise equal to layer-wise l2 norm of params
                noise = torch.randn_like(layer)
                scaling_factor = layer.grad.norm(p=2)/math.sqrt(layer.numel())
                # Multiply by constant scaling factor
                scaling_factor = scaling_factor*config['noise_scale_factor']
                layer.grad.data += noise*scaling_factor
                # Append to list for logging purposes
                noise_per_layer.append(scaling_factor)
            
            # Finally, mask gradient for pruned weights
            for prunable_layer, layer_mask in zip(self.prunable_params, self.mask):
                prunable_layer.grad.data *= layer_mask

        return noise_per_layer

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