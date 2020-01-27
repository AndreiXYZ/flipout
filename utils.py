import torch
import numpy as np
import gc
import torch.optim as optim
from datetime import datetime
from rmspropw import RMSpropW

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def get_time_str():
    now = datetime.now()
    return now.strftime('[%d-%m-%y %H:%M:%S]')

def get_opt(config, model):
    params = model.parameters()
    lr = config['lr']
    if config['reg_type'] == 'wdecay':
        wdecay = config['lambda']
    else:
        wdecay = 0
    
    if config['opt'] == 'adam':
        opt = optim.Adam(params, lr=lr, weight_decay=wdecay)
    elif config['opt'] == 'sgd':
        opt = optim.SGD(params, lr=lr, weight_decay=wdecay)
    elif config['opt'] == 'rmsprop':
        opt = optim.RMSprop(params, lr=lr, weight_decay=wdecay)
    elif config['opt'] == 'rmspropw':
        opt = RMSpropW(params, lr=lr, weight_decay=wdecay)
    return opt

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
            # elif 'bias' in name:
            #     writer.add_histogram('biases/'+name, layer.clone().detach().flatten(), epoch_num, bins=50)


def plot_stats(train_acc, train_loss, test_acc, test_loss, model, writer, epoch_num, config):
        writer.add_scalar('acc/train', train_acc, epoch_num)
        writer.add_scalar('acc/test', test_acc, epoch_num)
        writer.add_scalar('acc/generalization_err', train_acc-test_acc, epoch_num)
        writer.add_scalar('loss/train', train_loss, epoch_num)
        writer.add_scalar('loss/test', test_loss, epoch_num)
        writer.add_scalar('sparsity/sparsity', model.get_sparsity(config), epoch_num)
    
def print_gc_memory_usage():
    total_usage = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                total_usage += obj.element_size()*obj.nelement()
                print(type(obj), obj.size(), obj.requires_grad, obj.element_size()*obj.nelement())
        except:
            pass
    print('Total usage in bytes = ', total_usage)

def log_uniform(a, b, size):
    return torch.Tensor(np.random.uniform(np.log(a), np.log(b), size)).exp()

def save_run(config, model, opt, curr_epoch, fpath):
    save_dict = {'model_state_dict': model.state_dict(),
                 'opt_state_dict': opt.state_dict(),
                 'epoch': curr_epoch
                 }
    # Add the hparams used for the run to save_dict
    save_dict.update(config)
    torch.save(save_dict, fpath)