import torch
from models.mnist_models import LeNet_300_100

def get_sparsity(model):
    num_params_model = []

chkpt = torch.load('./chkpts/ln300_test.pt')
print(chkpt.keys())
model = LeNet_300_100()
model.load_state_dict(chkpt['model_state'])

opt = torch.optim.Adam(model.parameters())
opt.load_state_dict(chkpt['opt_state'])

model.mask = chkpt['mask']