import torch
import numpy as np
import gc
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import io

from datetime import datetime
from rmspropw import RMSpropW
from models.cifar10_models import *
from models.mnist_models import *
from master_model import MasterWrapper
from L0_reg.L0_models import L0LeNet5, L0MLP
from imageio import imread

def accuracy(out, y):
    preds = out.argmax(dim=1, keepdim=True).squeeze()
    correct = preds.eq(y).sum().item()
    return correct

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def get_time_str():
    now = datetime.now()
    return now.strftime('[%d-%m-%y %H:%M:%S]')


def load_model(config):
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

    else:
        penalty = 0

    return penalty

def plot_weight_histograms(model, writer, epoch_num):
    for name,layer in model.named_parameters():
        if layer.requires_grad:
            layer_histogram = layer.clone().detach().flatten()
            # Remove outliers
            deviation = (layer_histogram - layer_histogram.mean()).abs()
            layer_histogram = layer_histogram[deviation < 2*layer_histogram.std()]
            # Get only nonzeros for visibility
            layer = layer[layer!=0]
            if 'weight' in name:
                writer.add_histogram('weights/'+name, layer_histogram, epoch_num)


def plot_layerwise_sparsity(model, writer, epoch_num):
    layerwise_sparsity = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            total_params = module.weight.numel()
            total_pruned = (module.weight==0).sum().item() 

            if module.bias is not None:
                total_params += module.bias.numel()
                total_pruned += (module.bias==0).sum().item()
            layerwise_sparsity.append(float(total_pruned)/total_params)
    
    for idx, elem in enumerate(layerwise_sparsity):
        writer.add_scalar('layer_sparsity/'+str(idx), elem, epoch_num)

def plot_hparams(writer, config, train_acc, test_acc, train_loss, test_loss, sparsity):
    import copy

    metrics = {'train_acc': train_acc,
               'test_acc': test_acc,
               'train_loss': train_loss,
               'test_loss': test_loss,
               'sparsity': sparsity}

    hparams = copy.deepcopy(config)
    hparams['lambas'] = hparams['lambas'][0]

    # Correct types if it's the case
    if hparams['reg_type'] is None:
        hparams['reg_type'] = 'none'
    if hparams['milestones'] is None:
        hparams['milestones'] = 'none'
    else:
        hparams['milestones'] = str(hparams['milestones'])
    
    # Delete uninformative parameters
    del hparams['device']    
    del hparams['logdir']

    for k,v in hparams.items():
        print('{} - {} - {}'.format(k, v, type(v)))
    writer.add_hparams(hparams, metrics)


def plot_stats(train_acc, train_loss, test_acc, test_loss, model, writer, epoch_num, config, cls_module):
        writer.add_scalar('acc/train', train_acc, epoch_num)
        writer.add_scalar('acc/test', test_acc, epoch_num)
        writer.add_scalar('acc/generalization_err', train_acc-test_acc, epoch_num)
        writer.add_scalar('loss/train', train_loss, epoch_num)
        writer.add_scalar('loss/test', test_loss, epoch_num)
        writer.add_scalar('sparsity/sparsity', model.sparsity, epoch_num)
        writer.add_scalar('sparsity/remaining_connections', get_num_connections(cls_module), epoch_num)


def get_num_connections(module):
    # For a layer, returns how many neurons have been disconnected
    # To be used mainly for the classification layer
    # Num. of neurons is the rows so sum over rows
    sum_connections = module.weight.sum(dim=1)
    return (sum_connections!=0.).sum().item()

def save_run(config, model, opt, curr_epoch, fpath):
    save_dict = {'model_state_dict': model.state_dict(),
                 'opt_state_dict': opt.state_dict(),
                 'epoch': curr_epoch
                 }
    # Add the hparams used for the run to save_dict
    save_dict.update(config)
    torch.save(save_dict, fpath)