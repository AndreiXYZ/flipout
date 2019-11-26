import torch.nn as nn

class LeNet_300_100(nn.Module):
    def __init__(self):
        super(LeNet_300_100, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = x.flatten(start_dim=2)
        out = self.layers(x)
        
        return out