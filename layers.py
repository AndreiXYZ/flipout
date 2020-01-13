import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.nn.modules.utils import _pair

class LinearMasked(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(LinearMasked, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weights = nn.Parameter(torch.Tensor(output_features, input_features))

        if 'bias':
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        self.weight_signs = self.weights.clone().detach().sign().to('cuda')
        self.weights.data.abs_()

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            self.bias_signs = self.bias.clone().detach().sign().to('cuda')
            self.bias.data.abs_()
            
    def forward(self, input):
        signed_weights = F.relu(self.weights)*self.weight_signs
        
        if self.bias is not None:
            signed_bias = F.relu(self.bias)*self.bias_signs
        
        return F.linear(input, signed_weights, signed_bias)

class Conv2dMasked(nn.modules.conv._ConvNd):
    # TODO
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2dMasked, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight_signs = self.weight.clone().detach().sign().to('cuda')
        self.weight.data.abs_()

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            self.bias_signs = self.bias.clone().detach().sign().to('cuda')
            self.bias.data.abs_()

    def conv2d_forward(self, input, weight):
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
        
    def forward(self, input):
        return self.conv2d_forward(input, self.weight)