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
            weights.data = weights*layer_mask

    def update_mask(self, rate):
        # Prune parameters of the network according to a criterion
        # Criterion is a func by which the weights are sorted
        for idx, layer in enumerate(self.parameters()):
            flat_layer = layer.view(-1)
            num_pruned = (flat_layer==0).sum().item()
            num_elems = flat_layer.numel()
            num_unpruned = num_elems - num_pruned
            to_prune = num_pruned + int(rate*num_unpruned)

            inds = flat_layer.abs().argsort(descending=False)[:to_prune]

            new_mask = torch.ones(self.mask[idx].numel())
            new_mask[inds] = 0
            new_mask = new_mask.view_as(self.mask[idx])
            new_mask = new_mask.to(self.mask[idx].get_device())
            self.mask[idx] = self.mask[idx]*new_mask
            # curr_mask = self.mask[idx]

            # inds = inds[:to_prune]

            # self.mask[idx][inds] = 0
    def get_flips(self):
    # Retrieves how many params have flipped compared to previously saved weights
        num_flips = 0
        
        for curr_weights, prev_weights in zip(self.parameters(), self.saved_weights):
            curr_signs = curr_weights.sign()
            prev_signs = prev_weights.sign()
            flipped = ~curr_signs.eq(prev_signs)
            num_flips += flipped.sum()
        return num_flips
    
class LeNet_300_100(MasterModel):
    def __init__(self):
        super(LeNet_300_100, self).__init__()
        # Not exactly like the paper, yet
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
        # should maybe used self.named_parameters and check if weight is in 
        # name so as to exclude biases. are biases ok to prune?

    def forward(self, x):
        x = x.flatten(start_dim=2)
        out = self.layers(x)
        
        return out

class LeNet5(MasterModel):
    def __init__(self):
        super(LeNet5, self).__init__()

        # Not exactly like the paper, yet
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(6, 16, kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(16, 120, kernel_size=(5,5)),
            nn.ReLU()
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
        
    def forward(self, x):
        conv_out = self.conv_layers(x)
        # Flatten x
        conv_out = conv_out.flatten(start_dim=1)
        fc_out = self.fc_layers(conv_out)
        return fc_out