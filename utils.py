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


def plot_stats(train_acc, train_loss, test_acc, test_loss, model, writer, epoch_num):
        writer.add_scalar('acc/train', train_acc, epoch_num)
        writer.add_scalar('acc/test', test_acc, epoch_num)
        writer.add_scalar('loss/train', train_loss, epoch_num)
        writer.add_scalar('loss/test', test_loss, epoch_num)
        writer.add_scalar('sparsity/sparsity', model.get_sparsity(), epoch_num)
        # Visualise histogram of flips
        writer.add_histogram('layer 0 flips hist.', model.flip_counts[0].flatten(), epoch_num)


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
