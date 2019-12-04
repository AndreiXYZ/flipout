import torch
import numpy as np
from master_model import MasterWrapper, MasterModel
from models import *
from data_loaders import *

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def construct_run_name(config):
    return ''.join(['_'+str(key)+'_'+str(value) if key!='comment' else '' for key,value in config.items()])

def load_model(config):
    if config['model'] == 'lenet300':
        model = LeNet_300_100()
    elif config['model'] == 'lenet5':
        model = LeNet5() 
    
    model = MasterWrapper(model).to(config['device'])
    return model

def load_dataset(config):
    if config['dataset'] == 'mnist':
        train_loader, train_size, test_loader, test_size = get_mnist_loaders(config)
    elif config['dataset'] == 'cifar10':
        train_loader, train_size, test_loader, test_size = get_cifar10_loaders(config)
    
    return train_loader, train_size, test_loader, test_size
