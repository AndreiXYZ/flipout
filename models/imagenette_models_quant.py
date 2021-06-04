import torch
import torch.nn as nn
import torch.nn.functional as F
from .master_model import MasterModel
from torch.quantization import QuantStub, DeQuantStub
import torchvision.models


class DenseNet121Quant(MasterModel):
    def __init__(self, pretrained=False, num_classes=10):
        super(DenseNet121Quant, self).__init__()
        self.quant = QuantStub()
        self.model = torchvision.models.densenet121(pretrained=pretrained,
                                                    num_classes=num_classes)

        self.dequant = DeQuantStub()

    def forward(self, x):
        out = self.quant(x)
        out = self.model.forward(out)
        out = self.dequant(out)
        
        return out