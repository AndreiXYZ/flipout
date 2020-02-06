import torch.nn as nn
import torch
import math
import torchvision.models as models
from models.layers import LinearMasked, Conv2dMasked

def separate_signs(layer):
    layer.weight_signs = layer.weight.clone().detach().sign().to('cuda')
    layer.weight.data.abs_()

    if layer.bias is not None:
        layer.bias_signs = layer.bias.clone().detach().sign().to('cuda')
        layer.bias.data.abs_()

def selfmask_forward_linear(self, x):
    signed_weights = F.relu(self.weight)*self.weight_signs
    
    if self.bias is not None:
        signed_bias = F.relu(self.bias)*self.bias_signs
    
    return F.linear(x, signed_weights, signed_bias)

def selfmask_forward_conv2d(self, x):
    signed_weight = F.relu(weight)*self.weight_signs
        
    if self.bias is not None:
        signed_bias = F.relu(self.bias)*self.bias_signs
    
    if self.padding_mode == 'circular':
        expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                            (self.padding[0] + 1) // 2, self.padding[0] // 2)

        return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                        signed_weight, signed_bias, self.stride,
                        _pair(0), self.dilation, self.groups)

    return F.conv2d(input, signed_weight, signed_bias, self.stride,
                    self.padding, self.dilation, self.groups)

def convert_to_custom(model):
    for layer in model.modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            separate_signs(layer)
        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(selfmask_forward_linear, layer)
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(selfmask_forward_conv2d, layer)
