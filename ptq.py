import torch
import torch.nn as nn
import math
import argparse
import torch.nn.functional as F
import os
import logging
import utils.getters as getters
import utils.utils as utils
from tqdm import tqdm
from models.master_model import init_attrs, CustomDataParallel

activations = {}
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(name='ptq')

def load_weights_and_mask(config, model, model_state, mask):
    if config['model'] == 'densenet121quant':
        # Remove 'module' part of keys, as model was wrapped with dataparallel
        # when training
        model_state = {key.replace('module.',''):value for key,value in model_state.items()}

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
            total_acc += utils.accuracy(out, y)
            # multiply batch loss by batch size since the loss is averaged
            total_loss += x.size(0)*loss.item()
            total_samples += x.size(0)
            pbar.set_postfix({'acc' : total_acc / total_samples, 'loss' : total_loss / total_samples})
    
    total_acc /= total_samples
    total_loss /= total_samples

    return total_acc, total_loss

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    logger.info('Size (MB): {}'.format(os.path.getsize('temp.p')/1e6))
    os.remove('temp.p')


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook


def get_qconfig(config):
    weight_observer, weight_qscheme = getters.get_observer(config['weight_observer'], config['weight_qscheme'])
    activation_observer, activation_qscheme = getters.get_observer(config['activation_observer'], config['activation_qscheme'])

    logger.info('Weight: observer={} qscheme={}'.format(weight_observer, weight_qscheme))
    logger.info('Activation: observer={} qscheme={}'.format(activation_observer, activation_qscheme))

    weight_observer = weight_observer.with_args(dtype=torch.qint8,
                                                qscheme=weight_qscheme,
                                                reduce_range=False)

    activation_observer = activation_observer.with_args(dtype=torch.quint8,
                                                        qscheme=activation_qscheme,
                                                        reduce_range=False)
    
    return weight_observer, activation_observer


def prepare_model_for_quantization(model, config):
    model.eval()
    model.to('cpu')

    weight_observer, activation_observer = get_qconfig(config)

    model.qconfig = torch.quantization.QConfig(weight=weight_observer,
                                               activation=activation_observer)

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
    state_dict = utils.load_state_dict(fpath=config['saved_model_path'])
    model_state = state_dict['model_state']
    mask = state_dict['mask']
    training_cfg = state_dict['config']

    utils.set_seed(training_cfg['seed'])
    # Instantiate model & attributes, load state dict 
    model = getters.get_quant_model(config)

    # Need to wrap model with data parallel for it to work it seems

    init_attrs(model, training_cfg)

    # Load model weights and mask
    model = load_weights_and_mask(config, model, model_state, mask)
    print_size_of_model(model)
    # Switch to eval mode, move to cpu and prepare for quantization
    # do module fusion
    # fuse_model(model)
    quant_model = prepare_model_for_quantization(model, config)

    # Grab all necessary objects
    loaders, sizes = getters.get_dataloaders(training_cfg)
    train_loader, _, test_loader = loaders
    train_size, _, test_size = sizes
    batches_per_train_epoch = math.ceil(train_size / training_cfg['batch_size'])
    batches_per_test_epoch = math.ceil(test_size / training_cfg['test_batch_size'])

    # Calibration (could possibly use more epochs)
    calib_acc, calib_loss = evaluate(quant_model, train_loader, batches_per_train_epoch)

    torch.quantization.convert(quant_model, inplace=True)
    logger.info('Succesfully quantized model!')

    print_size_of_model(quant_model)
    logger.info('Evaluating...')
    train_acc, train_loss = evaluate(quant_model, train_loader, batches_per_train_epoch)
    test_acc, test_loss = evaluate(quant_model, test_loader, batches_per_test_epoch)
    
    logger.info('train acc: {} train loss: {}'.format(train_acc, train_loss))
    logger.info('test acc: {} test loss: {}'.format(test_acc, test_loss))
    
    # Save model in same folder
    save_path = config['saved_model_path'].replace('model.pt', '')
    utils.save_run_quant(quant_model, save_path, config, train_acc, test_acc)


if __name__ == "__main__":
    model_choices = ['vgg19quant', 'resnet18quant', 'densenet121quant']
    observer_choices = ['minmax', 'ma-minmax', 'pc-minmax', 'ma-pc-minmax', 'hist']
    qscheme_choices = ['affine', 'symmetric']
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, choices=model_choices, required=True)
    # parser.add_argument('-bs', '--batch_size', type=int, required=True)
    # parser.add_argument('-tbs', '--test_batch_size', type=int, required=True)
    parser.add_argument('--val', action='store_true', default=False)
    parser.add_argument('--prune_bias', action='store_true', default=False)
    parser.add_argument('--prune_bnorm', action='store_true', default=False)
    parser.add_argument('--noise_only_prunable', action='store_true', default=False)
    parser.add_argument('--saved_model_path', type=str, required=True)
    parser.add_argument('--weight_observer', type=str, choices=observer_choices, required=True)
    parser.add_argument('--weight_qscheme', type=str, choices=qscheme_choices, required=True)
    parser.add_argument('--activation_observer', type=str, choices=observer_choices, required=True)
    parser.add_argument('--activation_qscheme', type=str, choices=qscheme_choices, required=True)

    config = vars(parser.parse_args())
    main(config)