import torch
import numpy as np
import gc
import torch.optim as optim

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def construct_run_name(config):
    return ''.join(['_'+str(key)+'_'+str(value) if key!='comment' else '' for key,value in config.items()])

def get_opt(config, params):
    if config['opt'] == 'adam':
        opt = optim.Adam(params, lr=config['lr'], weight_decay=config['wdecay'])
    elif config['opt'] == 'sgd':
        opt = optim.SGD(params, lr=config['lr'], weight_decay=config['wdecay'])
    elif config['opt'] == 'rmsprop':
        opt = optim.RMSprop(params, lr=config['lr'], weight_decay=config['wdecay'])
    
    return opt
    
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
