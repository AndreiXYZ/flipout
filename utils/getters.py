import torch.optim as optim

from utils import data_loaders
from models.mnist_models import *
from models.cifar10_models import *
from models.cifar10_models_quant import *
from models.imagenette_models import *
from models.imagenette_models_quant import DenseNet121Quant

def get_model(config):
    init_param = 'VGG19' if config['model'] == 'vgg19' else None
    model_dict = {'lenet300': LeNet_300_100,
                  'lenet5': LeNet5,
                  'conv6': Conv6,
                  'vgg19': VGG,
                  'resnet18': ResNet18,
                  'densenet121': DenseNet121
                  }
    
    # Grab appropriate class and instantiate it
    if config['model'] == 'vgg19':
        model = VGG('VGG19')
    elif config['model'] == 'vgg16':
        model = VGG('VGG16')
    elif config['model'] == 'vgg13':
        model = VGG('VGG13')
    else:
        model = model_dict[config['model']]()

    if config['load_model']:
        raise NotImplementedError
        checkpoint = torch.load(config['load_model'], map_location='cuda')
        model.load_state_dict(checkpoint['model_state'])
    
    return model

def get_quant_model(config):
    model_dict = {'vgg19quant' : VGGQuant('VGG19'),
                  'resnet18quant' : ResNet18Quant(),
                  'densenet121quant' : DenseNet121Quant()}
    
    return model_dict[config['model']]

def get_observer(observer_type, qscheme_type):
    observer, qscheme = None, None

    assert observer_type in ('minmax', 'ma-minmax', 'pc-minmax', 'ma-pc-minmax', 'hist')
    assert qscheme_type in ('affine', 'symmetric')

    if observer_type == 'minmax':
        observer = torch.quantization.MinMaxObserver
    elif observer_type == 'ma-minmax':
        observer = torch.quantization.MovingAverageMinMaxObserver
    elif observer_type == 'pc-minmax':
        observer = torch.quantization.PerChannelMinMaxObserver
    elif observer_type == 'ma-pc-minmax':
        observer = torch.quantization.MovingAveragePerChannelMinMaxObserver
    elif observer_type == 'hist':
        observer = torch.quantization.HistogramObserver
    else:
        raise ValueError('Please specify a valid observer')


    if 'pc' in observer_type:
        if qscheme_type == 'affine':
            qscheme = torch.per_channel_affine
        elif qscheme_type == 'symmetric':
            qscheme = torch.per_channel_symmetric
    else:
        if qscheme_type == 'affine':
            qscheme = torch.per_tensor_affine
        elif qscheme_type == 'symmetric':
            qscheme = torch.per_tensor_symmetric

    return observer, qscheme

def get_dataloaders(config):
    if config['dataset'] == 'mnist':
        train_loader, test_loader = data_loaders.mnist_dataloaders(config)
    elif config['dataset'] == 'cifar10':
        loaders, sizes = data_loaders.cifar10_dataloaders(config)
    elif config['dataset'] == 'imagenette':
        loaders, sizes = data_loaders.imagenette_dataloaders(config)

    return loaders, sizes


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
                }
    
    opt = opt_dict[config['opt']](**kwargs)

    if config['load_model']:
        raise NotImplementedError
        checkpoint = torch.load(config['load_model'], map_location='cuda')
        opt.load_state_dict(checkpoint['opt_state'])
    
    return opt

def get_weight_penalty(model, config, epoch_num):
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
    else:
        penalty = 0
    
    penalty = penalty*config['lambda']

    hoyer_penalty = None
    if 'stop_hoyer_at' not in config or (
        'stop_hoyer_at' in config and epoch_num <= config['stop_hoyer_at']):
        if config['add_hs']:
            for layer in model.parameters():
                if layer.requires_grad and layer.abs().sum() > 0:
                    if hoyer_penalty is None:
                        hoyer_penalty = (layer.abs().sum()**2)/((layer.abs()**2).sum())
                    else:
                        hoyer_penalty += (layer.abs().sum()**2)/((layer.abs()**2).sum())        
        else:
            hoyer_penalty = 0
    else:
        hoyer_penalty = 0
    
    hoyer_penalty = hoyer_penalty*config['hoyer_lambda']

    return penalty + hoyer_penalty

def get_epoch_type(config):
    from utils.epoch_funcs import epoch_flips, regular_epoch

    if config['prune_criterion'] == 'flipout':
        return epoch_flips
        
    return regular_epoch