import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_mnist_loaders(config):
    
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    train_size = len(train_set)
    test_size = len(test_set)
    train_loader = DataLoader(train_set, 
                              batch_size = config['batch_size'],
                              shuffle = True, 
                              pin_memory = True,
                              num_workers = 8,
                              drop_last = False)
    test_loader = DataLoader(test_set,
                             batch_size = config['batch_size'],
                             shuffle=False,
                             pin_memory = True,
                             num_workers = 8,
                             drop_last = False)
    
    return train_loader, train_size, test_loader, test_size


def get_cifar10_loaders(config):
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    train_size = len(train_set)
    test_size = len(test_set)

    train_loader = DataLoader(train_set,
                              batch_size = config['batch_size'],
                              shuffle = True,
                              pin_memory = True,
                              num_workers = 8,
                              drop_last = False)

    test_loader = DataLoader(test_set,
                             batch_size = config['batch_size'],
                             shuffle = False,
                             pin_memory = True,
                             num_workers = 8,
                             drop_last = False)
    
    return train_loader, train_size, test_loader, test_size