import torch.nn as nn
import torch
import math
import torchvision.models as models
from layers import LinearMasked
from master_model import MasterModel, MasterWrapper

class LeNet300Custom(MasterModel):
    def __init__(self):
        super(LeNet300Custom, self).__init__()
        self.layers = nn.Sequential(LinearMasked(28*28, 300),
                                    nn.ReLU(),
                                    LinearMasked(300, 100),
                                    nn.ReLU(),
                                    LinearMasked(100,10))
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.layers(x)

class LeNet_300_100(MasterModel):
    def __init__(self):
        super(LeNet_300_100, self).__init__()
        # Not exactly like the paper, yet
        self.layers = nn.Sequential(
            nn.Linear(28*28, 300),
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
        x = x.view(-1, 28*28)
        out = self.layers(x)
        
        return out

class LeNet5(MasterModel):
    def __init__(self):
        super(LeNet5, self).__init__()

        # Not exactly like the paper, yet
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5,5)),
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

class ResNet18(MasterModel):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False)

    def forward(self, x):
        for module in self.model.modules():
            if type(module) is not nn.Sequential:
                print(module)
        breakpoint()
        out = self.model.features(x)
        print(out.shape)

class VGG11(MasterModel):
    def __init__(self):
        super(VGG11, self).__init__()
        self.model = models.vgg11(pretrained=False, num_classes=10)
    
    def forward(self, x):
        out = self.model.features(x)
        out = self.model.avgpool(out)
        out = out.flatten(start_dim=1)
        out = self.model.classifier(out)
        return out

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
    if config['model'] == 'lenet300':
        model = LeNet_300_100()
    elif config['model'] == 'lenet5':
        model = LeNet5() 
    elif config['model'] == 'resnet18':
        model = ResNet18()
    elif config['model'] == 'vgg11':
        model = VGG11()
    elif config['model'] == 'custom':
        model = LeNet300Custom()
    
    model = MasterWrapper(model).to(config['device'])

    return model