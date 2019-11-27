import torch
import numpy as np
import argparse
import torch.optim as optim

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
            opt.step()

        preds = out.argmax(dim=1, keepdim=True).squeeze()
        correct = preds.eq(y).sum().item()
        
        epoch_acc += correct
        epoch_loss += loss.item()
    
    epoch_acc /= size
    epoch_loss /= size

    return epoch_acc, epoch_loss 

def train(config):
    device = config['device']
    model = LeNet_300_100().to(device)
    opt = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()

    train_loader, train_size, test_loader, test_size = get_mnist_loaders(config)

    for epoch_num in range(config['epochs']):
        model.train()
        train_acc, train_loss = epoch(train_loader, train_size, model, opt, criterion, device, config)

        model.eval()
        with torch.no_grad():
            test_acc, test_loss = epoch(test_loader, test_size, model, opt, criterion, device, config)   

        print('Train - acc: {} loss: {}\nTest - acc: {} loss: {}'.format(
            train_acc, train_loss, test_acc, test_loss
        ))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    config = vars(parser.parse_args())

    # Ensure experiment is reproducible
    set_seed(config['seed'])


    train(config)

if __name__ == "__main__":
    main()