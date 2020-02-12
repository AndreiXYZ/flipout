import torch.nn as nn
import torch
import math
import torchvision.models as models
from models.layers import LinearMasked, Conv2dMasked

def separate_signs(module):
    # Separate signs from values of the weights
    # and make sure signs are not backpropagated through
    module.weight_sign = module.weight.sign()
    module.weight_sign.requires_grad = False
    module.weight.abs_()

    # Repeat for bias if it is the case
    if module.bias is not None:
        module.bias_sign = module.bias.sign()
        module.bias_sign.requires_grad = False
        module.bias.abs_()


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
