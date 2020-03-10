import torch.optim as optim

from utils import data_loaders
from models.cifar10_models import *
from models.mnist_models import *
from models.L0_models import *
from rmspropw import RMSpropW
from models.master_model import MasterWrapper

def get_model(config):
    init_param = 'VGG19' if config['model'] == 'vgg19' else None
    model_dict = {'lenet300': LeNet_300_100,
                  'lenet5': LeNet5,
                  'conv6': Conv6,
                  'vgg19': VGG,
                  'resnet18': ResNet18,
                  'l0lenet5': L0LeNet5,
                  'l0lenet300': L0MLP,
                  }
    
    # Grab appropriate class and instantiate it
    if config['model'] == 'vgg19':
        model = VGG('VGG19')

    elif 'l0' in config['model']:
        model = model_dict[config['model']](N=60000, weight_decay=config['lambda'],
                                            lambas=config['lambas'], local_rep=config['local_rep'],
                                            temperature=config['temperature'], beta_ema=config['beta_ema'])
    else:
        model = model_dict[config['model']]()
    # Now wrap it in the master wrapper class if we're doing flips
    if config['prune_criterion'] == 'flip':
        model = MasterWrapper(model).to(config['device'])
    
    return model

def get_dataloaders(config):
    if config['dataset'] == 'mnist':
        train_loader, test_loader = data_loaders.mnist_dataloaders(config)
    elif config['dataset'] == 'cifar10':
        train_loader, test_loader = data_loaders.cifar10_dataloaders(config)
    
    return train_loader, test_loader


def get_opt(config, model):
    lr = config['lr']
    if config['reg_type'] == 'wdecay':
        wdecay = config['lambda']
    else:
        wdecay = 0
    
    kwargs = {'params': model.parameters(),
              'lr': config['lr'], 
              'weight_decay': wdecay,
              }
    # Add momentum if opt is not Adam
    if config['opt'] != 'adam':
        kwargs['momentum'] = config['momentum']
    
    opt_dict = {'adam': optim.Adam,
                'sgd': optim.SGD,
                'rmsprop': optim.RMSprop,
                'rmspropw': RMSpropW}
    
    opt = opt_dict[config['opt']](**kwargs)
    return opt

def get_weight_penalty(model, config):
    if 'l0' in config['model']:
        return 0
    
    penalty = None
    if config['reg_type'] == 'l1':
        for layer in model.parameters():
            if penalty is None:
                penalty = layer.norm(p=1)
            else:
                penalty = penalty + layer.norm(p=1)

    elif config['reg_type'] == 'l2':
        for layer in model.parameters():
            if penalty is None:
                penalty = layer.norm(p=2)**2
            else:
                penalty = penalty + layer.norm(p=2)**2

    elif config['reg_type'] == 'hs':
        for layer in model.parameters():
            if penalty is None:
                penalty = layer.abs().norm(p=1)**2 / layer.abs().norm(p=2)**2
            else:
                penalty += layer.abs().norm(p=1)**2 / layer.abs().norm(p=2)**2
    
    else:
        penalty = 0

    return penalty

def get_epoch_type(config):
    from utils.epoch_funcs import epoch_l0, epoch_flips, regular_epoch

    if config['prune_criterion'] in ['flip', 'topflip', 'topflip_layer']:
        return epoch_flips
    elif config['prune_criterion'] == 'l0':
        return epoch_l0
        
    return regular_epoch