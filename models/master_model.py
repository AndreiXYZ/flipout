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
                                 if isinstance(module, prunable_modules)
                                 and module.weight.requires_grad
                                 and module.bias.requires_grad]
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

    if 'flip' in config['prune_criterion']:
        model.flip_counts = [torch.zeros_like(layer, dtype=torch.float).to('cuda') 
                            for layer in model.prunable_params]
                            
        model.ema_flip_counts = [torch.zeros_like(layer, dtype=torch.float).to('cuda') 
                                for layer in model.prunable_params]
    
    if config['prune_criterion'] == 'historical_magnitude':
        model.historical_magnitudes = [torch.zeros_like(layer, dtype=torch.float).to('cuda')
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

    def save_rewind_weights(self):
        self.rewind_weights = [layer.data.detach().clone().to('cuda') for layer in self.parameters()]

    def rewind(self):
        for weights, rewind_weights, layer_mask in zip(self.parameters(), self.rewind_weights, self.mask):
            weights.data = rewind_weights.data*layer_mask
    
    def mask_weights(self, config):
        with torch.no_grad():
            for weights, layer_mask in zip(self.prunable_params, self.mask):
                weights.data = weights.data*layer_mask
    
    
    def mask_grads(self, config):
        with torch.no_grad():
            for weights, layer_mask in zip(self.prunable_params, self.mask):
                weights.grad.data = weights.grad.data*layer_mask

    def mask_filter(self, layer, filter_num, config):
        # TODO this does not take into account residual connections
        # Prune the actual filter
        self.prunable_params[layer][filter_num, :, :, :] = 0.
        self.mask[layer][filter_num, :, :, :] = 0.
        # If pruning bias, prune bias corresponding to that filter
        # and then prune the corresponding channels in next layer
        if config['prune_bias']:
            assert len(self.prunable_params[layer+1].size()) == 1
            self.prunabe_modules[layer+1][filter_num] = 0.
            self.mask[layer+1][filter_num] = 0.
            # Prune channels of next layer only if it is not linear
            if len(self.prunable_params[layer+2].size()) == 4:
                self.prunable_params[layer+2][:, filter_num, :, :] = 0.
                self.mask[layer+2][:, filter_num, :, :] = 0.
        else:
            # Otherwise just prune corresponding channels in next layer
            if len(self.prunable_params[layer+1].size()) == 4:
                self.prunable_params[layer+1][:, filter_num, :, :] = 0.
                self.mask[layer+1][:, filter_num, :, :] = 0.

    def update_mask_structured_magnitudes(self, config):
        with torch.no_grad():
            all_units = []

            for idx_layer, (layer, layer_mask) in enumerate(zip(self.prunable_params, self.mask)):
                if len(layer.size()) == 4:
                    # size = (num_filters, num_channels, width, height)
                    all_units.extend([(idx_layer, idx_filt, filt.abs().mean()) 
                                    for idx_filt,filt in enumerate(layer.unbind(dim=0))])

            dtype = np.dtype([('layer_idx', np.int32), ('filt_idx', np.int32), 
                            ('value', torch.Tensor)])
            
            all_units_arr = np.array(all_units, dtype=dtype)
            idxs = np.argsort(all_units_arr, order='value')
            # Prune first 5 for testing purposes
            for idx_to_prune in idxs[:5]:
                layer_idx = all_units_arr[idx_to_prune][0]
                filt_idx = all_units_arr[idx_to_prune][1]
                self.mask_filter(layer_idx, filt_idx, config)

        
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
    
    def update_mask_historical_magnitudes(self, rate):
        with torch.no_grad():
            flat_params = torch.cat([layer.view(-1) for layer in self.prunable_params])
            flat_mask = torch.cat([layer_mask.view(-1) for layer_mask in self.mask])
            flat_historical_magnitudes = torch.cat([layer.view(-1) for layer 
                                                    in self.historical_magnitudes])

            num_pruned = (flat_params==0).sum().item()
            num_to_prune = int((self.total_prunable - num_pruned)*rate)
            to_prune = flat_historical_magnitudes.argsort(descending=False)[:num_pruned+num_to_prune]

            flat_mask[to_prune] = 0.
            self.mask = self.unflatten_tensor(flat_mask, self.mask)
            # Now update the weights and the counts
            for layer, layer_mask, historical_layer in zip(self.prunable_params, self.mask, self.historical_magnitudes):
                layer.data = layer * layer_mask
                historical_layer.data = historical_layer * layer_mask
        
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
    
    def update_mask_flips(self, threshold):
        with torch.no_grad():
        # Prune parameters based on sign flips
            for layer, layer_flips, layer_mask in zip(self.prunable_params, self.flip_counts, self.mask):
                # Get parameters whose flips are above a threshold and invert for masking
                flip_mask = ~(layer_flips >= threshold)
                layer_mask.data = flip_mask*layer_mask
                layer_flips.data *= layer_mask
                layer.data = layer*layer_mask
    

    def update_mask_topflips(self, rate, use_ema, reset_flips):
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
                                    for layer in self.prunable_params])
            
            # Get first n% flips
            num_pruned = (flat_mask==0).sum().item()
            num_to_prune = int((self.total_prunable - num_pruned)*rate)
            to_prune = flip_cts.argsort(descending=True)[:num_to_prune]
            # Grab only those who are different than 0
            to_prune = to_prune[flip_cts[to_prune] > 0]
            # Update mask and flip counts
            flat_mask[to_prune] = 0
            flip_cts[to_prune] = 0
            # Replace mask and update parameters
            self.mask = self.unflatten_tensor(flat_mask, self.mask)
            
            for layer, layer_mask in zip(self.prunable_params, self.mask):
                layer.data = layer*layer_mask

            # Reset flip counts after pruning
            if reset_flips:
                self.reset_flip_counts()
            
    def update_mask_topflips_layerwise(self, rate):
        with torch.no_grad():
            for layer, layer_mask, layer_flips in zip(self.prunable_params, self.mask, self.flip_counts):
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
    
    def update_mask_weight_div_flips(self, rate):
        with torch.no_grad():
            flat_mask = torch.cat([layer_mask.view(-1) for layer_mask in self.mask])
            flat_magnitudes = torch.cat([layer.view(-1) for layer in self.prunable_params])
            flip_cts = torch.cat([layer_flips.view(-1) for layer_flips in self.flip_counts])

            # Determine how many we need to prune
            num_pruned = (flat_mask==0).sum().item()
            num_to_prune = int((self.total_prunable-num_pruned)*rate)
            # Do weight divided by number of flips
            # add a 1 to denominator to avoid division by 0
            criterion = flat_magnitudes.abs()/(flip_cts+1)
            to_prune = criterion.argsort(descending=False)[:num_pruned+num_to_prune]
            flat_mask[to_prune] = 0.
            self.mask = self.unflatten_tensor(flat_mask, self.mask)

            for layer, layer_mask in zip(self.prunable_params, self.mask):
                layer.data = layer*layer_mask
    
    def update_mask_weight_squared_div_flips(self, rate):
        with torch.no_grad():
            flat_mask = torch.cat([layer_mask.view(-1) for layer_mask in self.mask])
            flat_magnitudes = torch.cat([layer.view(-1) for layer in self.prunable_params])
            flip_cts = torch.cat([layer_flips.view(-1) for layer_flips in self.flip_counts])

            # Determine how many we need to prune
            num_pruned = (flat_mask==0).sum().item()
            num_to_prune = int((self.total_prunable-num_pruned)*rate)

            # Do weight divided by number of flips
            # add a 1 to denominator to avoid division by 0
            criterion = flat_magnitudes.pow(2)/(flip_cts+1)
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
            
    def update_mask_sensitivity(self, sensitivity):
        # Updates the mask, removing all parameters that are less than std(layer)*sensitivity
        with torch.no_grad():
            for layer, layer_mask in zip(self.prunable_params, self.mask):
                threshold = sensitivity*layer.std()
                layer_mask.data = ~(layer.abs() < threshold)
                layer.data = layer*layer_mask
    
    def update_mask_threshold(self, threshold):
        # Prune all weights below a threshold
        with torch.no_grad():
            for layer, layer_mask in zip(self.prunable_params, self.mask):
                layer_mask.data = ~(layer.abs() < threshold)*layer_mask
                
                num_nonzeros = (layer_mask.view(-1)==0).sum()
                layer.data = layer*layer_mask
    
    def reset_flip_counts(self):
        for layer_flips, layer_ema_flips in zip(self.flip_counts, self.ema_flip_counts):
            layer_flips.data = torch.zeros_like(layer_flips)
            layer_ema_flips.data = torch.zeros_like(layer_ema_flips)
    
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


    def store_ema_flip_counts(self, beta):
        with torch.no_grad():
            for layer_flips, layer_ema_flips, layer_mask in zip(self.flip_counts, self.ema_flip_counts, self.mask):
                layer_ema_flips.data = beta*layer_ema_flips + (1-beta)*layer_flips
                layer_ema_flips.data = layer_ema_flips*layer_mask
    

    def add_current_magnitudes(self, config):
        with torch.no_grad():
            # Normalize the history every time if it is the case
            if config['normalize_magnitudes']:
                params = torch.cat([layer.clone().view(-1).abs() for layer in self.prunable_params])
                params = params/params.sum()
                params_to_add = self.unflatten_tensor(params, self.prunable_params)
            else:
                params_to_add = self.prunable_params
            

            for layer, historical_layer in zip(params_to_add, 
                                               self.historical_magnitudes):
                if config['beta_ema_maghists'] is not None:
                    historical_layer.data = historical_layer*config['beta_ema_maghists'] + layer.abs()
                else:
                    historical_layer.data = historical_layer*config['beta_ema_maghists'] + layer.abs()

    
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
            if config['global_noise']:
                flat_grads = torch.cat([layer.grad.view(-1) 
                                            for layer in self.noisy_params])
                
                scaling_factor = flat_grads.norm(p=2)/math.sqrt(flat_grads.numel())
                # Scale noise by lr if it's the case
                if config['scale_noise_by_lr']:
                    scaling_factor *= config['lr']/curr_lr
                # Multiply scaling factor by constant
                scaling_factor = scaling_factor*config['noise_scale_factor']
                
                for layer in self.noisy_params:
                    noise = torch.randn_like(layer)
                    layer.grad.data += noise*scaling_factor
                
                noise_per_layer.append(scaling_factor)

            else:
                for layer in self.noisy_params:
                    # Add noise equal to layer-wise l2 norm of params
                    noise = torch.randn_like(layer)
                    scaling_factor = layer.grad.norm(p=2)/math.sqrt(layer.numel())
                    # Scale noise by LR
                    if config['scale_noise_by_lr']:
                        scaling_factor *= config['lr']/curr_lr
                    # Multiply by constant scaling factor
                    scaling_factor = scaling_factor*config['noise_scale_factor']
                    layer.grad.data += noise*scaling_factor
                    # Append to list for logging purposes
                    noise_per_layer.append(scaling_factor)
            
            # Finally, mask gradient for pruned weights
            for prunable_layer, layer_mask in zip(self.prunable_params, self.mask):
                prunable_layer.grad.data *= layer_mask

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