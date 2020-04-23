import torch
import torch.nn as nn
import torch.nn.functional as F
from models.master_model import MasterModel

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