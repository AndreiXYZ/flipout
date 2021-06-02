import torch
import torch.nn as nn
import torch.nn.functional as F
from .master_model import MasterModel
import torchvision.models

class DenseNet121(MasterModel):
    def __init__(self, pretrained=False, num_classes=10):
        super(DenseNet121, self).__init__()
        self.model = torchvision.models.densenet121(pretrained=pretrained,
                                                    num_classes=num_classes)
    
    def forward(self, x):
        return self.model.forward(x)