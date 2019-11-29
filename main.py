import torch, argparse
import numpy as np
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from utils import *
from models import *
from data_loaders import *

def epoch(loader, size, model, opt, criterion, device, config):
    epoch_acc = 0
    epoch_loss = 0

    for x,y in loader:
        
        opt.zero_grad()

        x = x.float().to(device)
        y = y.to(device)

        out = model.forward(x).squeeze()
        loss = criterion(out, y)
        if model.training:
            loss.backward()
            model.apply_mask()
            opt.step()
        preds = out.argmax(dim=1, keepdim=True).squeeze()
        correct = preds.eq(y).sum().item()
        
        epoch_acc += correct
        epoch_loss += loss.item()
    
    epoch_acc /= size
    epoch_loss /= size

    return epoch_acc, epoch_loss 

def train(config):
    
    comment = construct_run_name(config)
    writer = SummaryWriter(comment=comment)

    device = config['device']

    if config['model'] == 'lenet300':
        model = LeNet_300_100()
    elif config['model'] == 'lenet5':
        model = LeNet5()

    model = MasterWrapper(model).to(device)
    print('Model has {} total params, including biases.'.format(model.get_total_params()))

    opt = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()

    train_loader, train_size, test_loader, test_size = get_mnist_loaders(config)

    for epoch_num in range(config['epochs']):
        print('='*10 + ' Epoch ' + str(epoch_num) + ' ' + '='*10)

        model.train()
        train_acc, train_loss = epoch(train_loader, train_size, model, opt, criterion, device, config)

        flips = model.get_flips()

        model.eval()
        with torch.no_grad():
            test_acc, test_loss = epoch(test_loader, test_size, model, opt, criterion, device, config)   

        print('Train - acc: {:>20} loss: {:>20}\nTest - acc: {:>21} loss: {:>21}'.format(
            train_acc, train_loss, test_acc, test_loss
        ))

        model.update_mask(0.2)
        print(model.get_sparsity())
        
        writer.add_scalar('acc/train', train_acc, epoch_num)
        writer.add_scalar('acc/test', test_acc, epoch_num)
        writer.add_scalar('loss/train', train_loss, epoch_num)
        writer.add_scalar('loss/test', test_loss, epoch_num)
        writer.add_scalar('flips/absolute', flips,epoch_num)
        writer.add_scalar('flips/percentage', float(flips)/model.total_params, epoch_num)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['lenet300', 'lenet5'], default='lenet300')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'], default='mnist')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    # Pruning
    parser.add_argument('--prune_rate', type=float, default=0.2)
    parser.add_argument('--prune_freq', type=int, default=2)

    config = vars(parser.parse_args())

    # Ensure experiment is reproducible.
    # Results may vary across machines!
    set_seed(config['seed'])


    train(config)

if __name__ == "__main__":
    main()