import torch.nn as nn
import torch
import numpy as np
from collections import OrderedDict

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


def _conv2d_flops(module, activation):
    # Auxiliary func to use abstract flop computation

    # Drop batch & channels. Channels can be dropped since
    # unlike shape they have to match to in_channels
    input_shape = activation.shape[2:]
    # TODO Add support for dilation and padding size
    total_flops = conv2d_flops(in_channels=module.in_channels,
                        out_channels=module.out_channels,
                        input_shape=input_shape,
                        kernel_shape=module.kernel_size,
                        padding=module.padding_mode,
                        strides=module.stride,
                        dilation=module.dilation)
    nonzero_in, nonzero_out = get_nonzeros(module)
    nonzero_flops = conv2d_flops(in_channels=nonzero_in,
                        out_channels=nonzero_out,
                        input_shape=input_shape,
                        kernel_shape=module.kernel_size,
                        padding=module.padding_mode,
                        strides=module.stride,
                        dilation=module.dilation)
    
    return total_flops, nonzero_flops

def _linear_flops(module, activation):
    # Auxiliary func to use abstract flop computation
    total_flops = dense_flops(module.in_features, module.out_features)
    nonzero_in, nonzero_out = get_nonzeros(module)
    nonzero_flops = dense_flops(nonzero_in, nonzero_out)

    return total_flops, nonzero_flops


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
    breakpoint
    activations = OrderedDict()

    def store_activations(module, input, output):
        if isinstance(module, nn.ReLU):
            # TODO ResNet18 implementation reuses a
            # single ReLU layer?
            return
        assert module not in activations, \
            f"{module} already in activations"
        # TODO [0] means first input, not all models have a single input
        if isinstance(input[0], list):
            inp = torch.cat(input[0], 1)
        else:
            inp = input[0]
        
        activations[module] = (inp.detach().cpu().numpy().copy(),
                                output.detach().cpu().numpy().copy(),)

    fn, hooks = hook_applyfn(store_activations, model, forward=True)
    model.apply(fn)
    with torch.no_grad():
        model(input)

    for h in hooks:
        h.remove()

    return activations


def get_flops(model, input):
    """Compute Multiply-add FLOPs estimate from model
    Arguments:
        model {torch.nn.Module} -- Module to compute flops for
        input {torch.Tensor} -- Input tensor needed for activations
    Returns:
        tuple:
        - int - Number of total FLOPs
        - int - Number of FLOPs related to nonzero parameters
    """

    # TODO this only works if, when pruning a filter,
    # I also prune the corresponding channels in the next
    # layer
    FLOP_fn = {
        nn.Conv2d: _conv2d_flops,
        nn.Linear: _linear_flops
    }

    total_flops = nonzero_flops = 0
    activations = get_activations(model, input)

    # The ones we need for backprop
    for m, (act, _) in activations.items():
        if m.__class__ in FLOP_fn:
            w = m.weight.detach().cpu().numpy().copy()
            total_flops_module, nonzero_flops_module = FLOP_fn[m.__class__](m, act)
            total_flops += total_flops_module
            nonzero_flops += nonzero_flops_module
        
    return total_flops, nonzero_flops


def get_nonzeros(module):
    assert (hasattr(module, 'weight')), 'The passed module has no weight parameter.'

    weight = module.weight.detach().cpu().numpy().copy()

    if isinstance(module, nn.Conv2d):
        dim0 = np.sum(np.sum(weight, axis=0),axis=(1,2))
        dim1 = np.sum(np.sum(weight, axis=1),axis=(1,2))
    if isinstance(module, nn.Linear):
        dim0 = np.sum(weight, axis=0)
        dim1 = np.sum(weight, axis=1)
    nz_count0 = np.count_nonzero(dim0)
    nz_count1 = np.count_nonzero(dim1)

    return nz_count0, nz_count1