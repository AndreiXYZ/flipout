import torch
import numpy as np
import gc
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from datetime import datetime
from models.cifar10_models import *
from models.mnist_models import *

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

def get_num_connections(module):
    # For a layer, returns how many neurons have been disconnected
    # To be used mainly for the classification layer
    # Num. of neurons is the rows so sum over rows
    sum_connections = module.weight.sum(dim=1)
    return (sum_connections!=0.).sum().item()

def plot_stats(train_acc, train_loss, test_acc, test_loss, model, writer, epoch_num, config):
        writer.add_scalar('acc/train', train_acc, epoch_num)
        writer.add_scalar('acc/test', test_acc, epoch_num)
        writer.add_scalar('acc/generalization_err', train_acc-test_acc, epoch_num)
        writer.add_scalar('loss/train', train_loss, epoch_num)
        writer.add_scalar('loss/test', test_loss, epoch_num)
        writer.add_scalar('sparsity/sparsity', model.sparsity, epoch_num)

def save_run(model, opt, config):
    import os

    save_dict = {
                 'opt_state': opt.state_dict(),
                 'model_state': model.state_dict(),
                 'mask': model.mask,
                 }
    
    save_fpath = './chkpts/' + config['logdir'] + '/' + config['save_model'] + '.pt'
    
    save_dir = '/'.join(save_fpath.split('/')[:-1])
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, 0o777)
    
    torch.save(save_dict, save_fpath)

def print_nonzeros(model):
    nonzero = total = 0
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        tensor = module.weight.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        # print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
        tensor = np.abs(tensor)
        if isinstance(module, nn.Conv2d):
            dim0 = np.sum(np.sum(tensor, axis=0),axis=(1,2))
            dim1 = np.sum(np.sum(tensor, axis=1),axis=(1,2))
        if isinstance(module, nn.Linear):
            dim0 = np.sum(tensor, axis=0)
            dim1 = np.sum(tensor, axis=1)
        nz_count0 = np.count_nonzero(dim0)
        nz_count1 = np.count_nonzero(dim1)
        print(f'{name:20} | dim0 = {nz_count0:7} / {len(dim0):7} ({100 * nz_count0 / len(dim0):6.2f}%) | dim1 = {nz_count1:7} / {len(dim1):7} ({100 * nz_count1 / len(dim1):6.2f}%)')