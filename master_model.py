import torch.nn as nn
import torch
import math

class MasterWrapper(object):
    def __init__(self, obj):
        self.obj = obj
        self.obj.total_params = self.obj.get_total_params()
        self.obj.pruned = 0
        self.obj.sparsity = 0
        self.obj.save_weights()
        self.obj.instantiate_mask()

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
    
    def save_weights(self):
        self.saved_weights = [weights.clone().detach().to('cuda')
                                for weights in self.parameters()]
    
    def instantiate_mask(self):
        self.mask = [torch.ones_like(weights).to('cuda')
                        for weights in self.parameters()]
    
    def apply_mask(self):
        for weights, layer_mask in zip(self.parameters(), self.mask):
            weights.grad.data = weights.grad.data*layer_mask
            weights.data = weights.data*layer_mask

    def update_mask(self, rate):
        # Prune parameters of the network according to a criterion
        # Criterion is a func by which the weights are sorted
        for layer, layer_mask in zip(self.parameters(), self.mask):
            flat_layer = layer.view(-1)

            num_pruned = (layer_mask==0).sum().item()
            num_unpruned = layer_mask.numel() - num_pruned
            to_prune = num_pruned + int(rate*num_unpruned)

            inds = flat_layer.abs().argsort(descending=False)[:to_prune]
            mask = layer_mask.view(-1).clone()
            mask[inds] = 0
            layer_mask.data = mask.view_as(layer_mask)
            layer.data = layer.data*layer_mask
    

    def get_sparsity(self):
        # Get the global sparsity rate
        sparsity = 0
        for layer in self.parameters():
            sparsity += (layer==0).sum().item()
        return float(sparsity)/self.total_params

    def get_flips(self):
    # Retrieves how many params have flipped compared to previously saved weights
        num_flips = 0
        
        for curr_weights, prev_weights in zip(self.parameters(), self.saved_weights):
            curr_signs = curr_weights.sign()
            prev_signs = prev_weights.sign()
            flipped = ~curr_signs.eq(prev_signs)
            num_flips += flipped.sum()
        return num_flips