import torch
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def construct_run_name(config):
    return ''.join(['_'+str(key)+'_'+str(value) if key!='comment' else '' for key,value in config.items()])