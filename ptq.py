import torch
import torch.nn as nn
import math
import argparse
import torch.nn.functional as F
import os
from models.cifar10_models_quant import ResNet18Quant
from utils.data_loaders import cifar10_dataloaders
from tqdm import tqdm
from utils.utils import load_state_dict, accuracy
from utils.getters import get_quant_model, get_epoch_type, get_opt
from models.master_model import init_attrs

activations = {}

def load_weights_and_mask(model, model_state, mask):
    model.load_state_dict(model_state)
    model.mask = mask
    return model


@torch.no_grad()
def evaluate(model, dataloader, total_batches):
    # Mention in paper this is taken from 
    total_loss = 0
    total_acc = 0
    total_samples = 0
    model.eval()
    pbar = tqdm(dataloader, total=total_batches)
    with torch.no_grad():
        for x,y in pbar:
            out = model(x)
            # Note this does not take into account the weight penalty
            loss = F.cross_entropy(out, y)
            total_acc += accuracy(out, y)
            # multiply batch loss by batch size since the loss is averaged
            total_loss += x.size(0)*loss.item()
            total_samples += x.size(0)
            pbar.set_postfix({'acc' : total_acc / total_samples, 'loss' : total_loss / total_samples})
    
    total_acc /= total_samples
    total_loss /= total_samples

    return total_acc, total_loss

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook


def prepare_model_for_quantization(model):
    model.eval()
    model.to('cpu')

    quantization_config = torch.quantization.QConfig(weight=torch.quantization.PerChannelMinMaxObserver.with_args(
                                                            dtype=torch.qint8, 
                                                            qscheme=torch.per_channel_symmetric,
                                                            reduce_range=False
                                                    ),
                                                     activation=torch.quantization.MinMaxObserver.with_args(
                                                            dtype=torch.quint8,
                                                            qscheme=torch.per_tensor_affine,
                                                            reduce_range=False
                                                     ))

    model.qconfig = quantization_config

    quant_model = torch.quantization.prepare(model)

    return quant_model

    
def fuse_model(model):
    named_modules = list(model.named_modules())
    
    for idx, named_module in enumerate(named_modules):
        conv_name, module = named_module
        if isinstance(module, nn.Conv2d):
            bnorm_name = named_modules[idx + 1][0]
            relu_name = named_modules[idx + 2][0]
            torch.quantization.fuse_modules(model, [conv_name, bnorm_name, relu_name], inplace=True)


def main(config):
    state_dict = load_state_dict(fpath=config['saved_model_path'])
    opt_state = state_dict['opt_state']
    model_state = state_dict['model_state']
    mask = state_dict['mask']
    training_cfg = state_dict['config']

    # Instantiate model & attributes, load state dict 
    model = get_quant_model(config)
    init_attrs(model, training_cfg)
    # Load model weights and mask
    model = load_weights_and_mask(model, model_state, mask)

    # Switch to eval mode, move to cpu and prepare for quantization
    # do module fusion
    # fuse_model(model)
    quant_model = prepare_model_for_quantization(model)

    # Grab all necessary objects
    epoch = get_epoch_type(training_cfg)

    loaders, sizes = cifar10_dataloaders(config)
    train_loader, _, test_loader = loaders
    train_size, _, test_size = sizes
    batches_per_train_epoch = math.ceil(train_size / config['batch_size'])
    batches_per_test_epoch = math.ceil(test_size / config['test_batch_size'])

    opt = get_opt(training_cfg, quant_model)

    # Calibration (could possibly use more epochs)
    calib_acc, calib_loss = evaluate(quant_model, train_loader, batches_per_train_epoch)

    torch.quantization.convert(quant_model, inplace=True)
    print('Quantized model size : {}'.format(print_size_of_model(quant_model)))
    train_acc, train_loss = evaluate(quant_model, train_loader, batches_per_train_epoch)
    test_acc, test_loss = evaluate(quant_model, test_loader, batches_per_test_epoch)
    
    print(train_acc, train_loss)
    print(test_acc, test_loss)
    
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    model_choices = ['vgg19quant', 'resnet18quant']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, choices=model_choices, required=True)
    parser.add_argument('-bs', '--batch_size', type=int, required=True)
    parser.add_argument('-tbs', '--test_batch_size', type=int, required=True)
    parser.add_argument('--val', action='store_true', default=False)
    parser.add_argument('--prune_bias', action='store_true', default=False)
    parser.add_argument('--prune_bnorm', action='store_true', default=False)
    parser.add_argument('--noise_only_prunable', action='store_true', default=False)
    parser.add_argument('--saved_model_path', type=str, required=True)

    config = vars(parser.parse_args())
    main(config)