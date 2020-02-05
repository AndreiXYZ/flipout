import torch.nn as nn
import torch
import math
import torchvision.models as models
from layers import LinearMasked, Conv2dMasked
from master_model import MasterModel, MasterWrapper

class LeNet300Custom(MasterModel):
    def __init__(self):
        super(LeNet300Custom, self).__init__()
        self.layers = nn.Sequential(LinearMasked(32*32, 300),
                                    nn.ReLU(),
                                    LinearMasked(300, 100),
                                    nn.ReLU(),
                                    LinearMasked(100,10))
        
    def forward(self, x):
        x = x.view(-1, 32*32)
        return self.layers(x)

class LeNet_300_100(MasterModel):
    def __init__(self):
        super(LeNet_300_100, self).__init__()
        # Not exactly like the paper, yet
        self.layers = nn.Sequential(
            nn.Linear(32*32, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

        # for layer in self.layers:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.xavier_normal_(layer.weight)
        # should maybe used self.named_parameters and check if weight is in 
        # name so as to exclude biases. are biases ok to prune?

    def forward(self, x):
        x = x.view(-1, 32*32)
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

class LeNet5Custom(MasterModel):
    def __init__(self):
        super(LeNet5Custom, self).__init__()

        # Not exactly like the paper, yet
        self.conv_layers = nn.Sequential(
            Conv2dMasked(in_channels=3, out_channels=6, kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            Conv2dMasked(6, 16, kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            Conv2dMasked(16, 120, kernel_size=(5,5)),
            nn.ReLU()
        )

        self.fc_layers = nn.Sequential(
            LinearMasked(120, 84),
            nn.ReLU(),
            LinearMasked(84, 10)
        )
        
    def forward(self, x):
        conv_out = self.conv_layers(x)
        # Flatten x
        conv_out = conv_out.flatten(start_dim=1)
        fc_out = self.fc_layers(conv_out)
        return fc_out

class Conv6(MasterModel):
    def __init__(self):
        super(Conv6, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=4096, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=10)
        )

    def forward(self, x):
        conv_out = self.conv_layers(x)
        conv_out = conv_out.flatten(start_dim=1)
        fc_out = self.fc_layers(conv_out)
        return fc_out

class Conv6Custom(MasterModel):
    def __init__(self):
        super(Conv6Custom, self).__init__()
        self.conv_layers = nn.Sequential(
            Conv2dMasked(in_channels=3, out_channels=64, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            Conv2dMasked(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv2dMasked(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            Conv2dMasked(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv2dMasked(in_channels=128, out_channels=256, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            Conv2dMasked(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            LinearMasked(4096, 256),
            nn.ReLU(),
            LinearMasked(256, 256),
            nn.ReLU(),
            LinearMasked(256, 10)
        )

    def forward(self, x):
        conv_out = self.conv_layers(x)
        conv_out = conv_out.flatten(start_dim=1)
        fc_out = self.fc_layers(conv_out)
        return fc_out

class ResNet18(MasterModel):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False, num_classes=10)

    def forward(self, x):
        out = self.model.conv1(x)
        out = self.model.bn1(out)
        out = self.model.relu(out)
        out = self.model.maxpool(out)
        out = self.model.layer1(out)
        out = self.model.layer2(out)
        out = self.model.layer3(out)
        out = self.model.layer4(out)
        out = self.model.avgpool(out)
        out = out.flatten(start_dim=1)
        out = self.model.fc(out)
        
        return out

def load_model(config):
    import time
    t1 = time.time()
    model_dict = {'lenet300': LeNet_300_100,
                  'lenet5': LeNet5,
                  'resnet18': ResNet18,
                  'conv6': Conv6,
                  'lenet300custom': LeNet300Custom,
                  'lenet5custom': LeNet5Custom,
                  'conv6custom': Conv6Custom}
    # Grab appropriate class and instantiate it
    model = model_dict[config['model']]()
    # Now wrap it in the master wrapper class if we're doing flips
    if config['prune_criterion'] == 'flip':
        model = MasterWrapper(model).to(config['device'])
    
    print('Time elapsed for load_model:', time.time() - t1)
    return model

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
