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



def dense_flops(in_neurons, out_neurons):
    """Compute the number of multiply-adds used by a Dense (Linear) layer"""
    return in_neurons * out_neurons


def conv2d_flops(in_channels, out_channels, input_shape, kernel_shape,
                 padding='same', strides=1, dilation=1):
    """Compute the number of multiply-adds used by a Conv2D layer
    Args:
        in_channels (int): The number of channels in the layer's input
        out_channels (int): The number of channels in the layer's output
        input_shape (int, int): The spatial shape of the rank-3 input tensor
        kernel_shape (int, int): The spatial shape of the rank-4 kernel
        padding ({'same', 'valid'}): The padding used by the convolution
        strides (int) or (int, int): The spatial stride of the convolution;
            two numbers may be specified if it's different for the x and y axes
        dilation (int): Must be 1 for now.
    Returns:
        int: The number of multiply-adds a direct convolution would require
        (i.e., no FFT, no Winograd, etc)
    """
    # validate + sanitize input
    assert in_channels > 0
    assert out_channels > 0
    assert len(input_shape) == 2
    assert len(kernel_shape) == 2
    padding = padding.lower()
    assert padding in ('same', 'valid', 'zeros'), "Padding must be one of same|valid|zeros"
    try:
        strides = tuple(strides)
    except TypeError:
        # if one number provided, make it a 2-tuple
        strides = (strides, strides)
    assert dilation == 1 or all(d == 1 for d in dilation), "Dilation > 1 is not supported"

    # compute output spatial shape
    # based on TF computations https://stackoverflow.com/a/37674568
    if padding in ['same', 'zeros']:
        out_nrows = np.ceil(float(input_shape[0]) / strides[0])
        out_ncols = np.ceil(float(input_shape[1]) / strides[1])
    else:  # padding == 'valid'
        out_nrows = np.ceil((input_shape[0] - kernel_shape[0] + 1) / strides[0])  # noqa
        out_ncols = np.ceil((input_shape[1] - kernel_shape[1] + 1) / strides[1])  # noqa
    output_shape = (int(out_nrows), int(out_ncols))

    # work to compute one output spatial position
    nflops = in_channels * out_channels * int(np.prod(kernel_shape))

    # total work = work per output position * number of output positions
    return nflops * int(np.prod(output_shape))

def nonzero(tensor):
    """Returns absolute number of values different from 0
    Arguments:
        tensor {numpy.ndarray} -- Array to compute over
    Returns:
        int -- Number of nonzero elements
    """
    return np.sum(tensor != 0.0)


def _conv2d_flops(module, activation):
    # Auxiliary func to use abstract flop computation

    # Drop batch & channels. Channels can be dropped since
    # unlike shape they have to match to in_channels
    input_shape = activation.shape[2:]
    # TODO Add support for dilation and padding size
    return conv2d_flops(in_channels=module.in_channels,
                        out_channels=module.out_channels,
                        input_shape=input_shape,
                        kernel_shape=module.kernel_size,
                        padding=module.padding_mode,
                        strides=module.stride,
                        dilation=module.dilation)


def _linear_flops(module, activation):
    # Auxiliary func to use abstract flop computation
    return dense_flops(module.in_features, module.out_features)


def hook_applyfn(hook, model, forward=False, backward=False):
    """
    [description]
    Arguments:
        hook {[type]} -- [description]
        model {[type]} -- [description]
    Keyword Arguments:
        forward {bool} -- [description] (default: {False})
        backward {bool} -- [description] (default: {False})
    Returns:
        [type] -- [description]
    """
    assert forward ^ backward, \
        "Either forward or backward must be True"
    hooks = []

    def register_hook(module):
        if (
            not isinstance(module, nn.Sequential)
            and
            not isinstance(module, nn.ModuleList)
            and
            not isinstance(module, nn.ModuleDict)
            and
            not (module == model)
        ):
            if forward:
                hooks.append(module.register_forward_hook(hook))
            if backward:
                hooks.append(module.register_backward_hook(hook))

    return register_hook, hooks


def get_activations(model, input):

    activations = OrderedDict()

    def store_activations(module, input, output):
        if isinstance(module, nn.ReLU):
            # TODO ResNet18 implementation reuses a
            # single ReLU layer?
            return
        assert module not in activations, \
            f"{module} already in activations"
        # TODO [0] means first input, not all models have a single input
        activations[module] = (input[0].detach().cpu().numpy().copy(),
                               output.detach().cpu().numpy().copy(),)

    fn, hooks = hook_applyfn(store_activations, model, forward=True)
    model.apply(fn)
    with torch.no_grad():
        model(input)

    for h in hooks:
        h.remove()

    return activations


def flops(model, input):
    """Compute Multiply-add FLOPs estimate from model
    Arguments:
        model {torch.nn.Module} -- Module to compute flops for
        input {torch.Tensor} -- Input tensor needed for activations
    Returns:
        tuple:
        - int - Number of total FLOPs
        - int - Number of FLOPs related to nonzero parameters
    """
    FLOP_fn = {
        nn.Conv2d: _conv2d_flops,
        nn.Linear: _linear_flops,
        Conv2dMasked: _conv2d_flops,
        LinearMasked: _linear_flops,
    }

    total_flops = nonzero_flops = 0
    activations = get_activations(model, input)

    # The ones we need for backprop
    for m, (act, _) in activations.items():
        if m.__class__ in FLOP_fn:
            w = m.weight.detach().cpu().numpy().copy()
            module_flops = FLOP_fn[m.__class__](m, act)
            total_flops += module_flops
            # For our operations, all weights are symmetric so we can just
            # do simple rule of three for the estimation
            nonzero_flops += module_flops * nonzero(w).sum() / np.prod(w.shape)

    return total_flops, nonzero_flops