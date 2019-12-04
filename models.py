import torch.nn as nn
import torch
import math
from master_model import MasterModel

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