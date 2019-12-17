import torch.nn as nn
import math
import torch.functional as F

class LinearMasked(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(LinearMasked, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(input_features, output_features))

        if 'bias':
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight_signs = self.weight.clone().detach().sign()
        self.weight.abs_()

        if self.bias is not None:
            fan_in, _ = nn.init.calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            self.bias_signs = self.bias.clone().detach().sign()
            self.bias.abs_()
            
    def forward(self, input):
        signed_weights = F.relu(self.weights)*self.weight_signs
        
        if self.bias is not None:
            signed_bias = F.relu(self.bias)*self.bias_signs
        
        return F.linear(input, signed_weights, signed_bias)
