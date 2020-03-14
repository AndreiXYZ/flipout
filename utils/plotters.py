import utils.utils as utils
import torch.nn as nn

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
    for name,module in model.named_modules():
        if hasattr(module, 'weight'):
            total_w = module.weight.numel()
            w_pruned = (module.weight==0).sum().item()
            writer.add_scalar('layerwise_sparsity/' + name + '.weight', float(w_pruned)/total_w, epoch_num)
        
        if hasattr(module, 'bias') and module.bias is not None:
            total_b = module.bias.numel()
            b_pruned = (module.bias==0).sum().item()
            writer.add_scalar('layerwise_sparsity/' + name + '.bias', float(b_pruned)/total_b, epoch_num)



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
        writer.add_scalar('sparsity/remaining_connections', utils.get_num_connections(cls_module), epoch_num)