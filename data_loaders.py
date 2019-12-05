import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_mnist_loaders(config):
    
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

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
    
    return train_loader, test_loader


def get_cifar10_loaders(config):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                                                        normalize]))
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                                                        normalize]))

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
    
    return train_loader, test_loader


def load_dataset(config):
    if config['dataset'] == 'mnist':
        train_loader, test_loader = get_mnist_loaders(config)
    elif config['dataset'] == 'cifar10':
        train_loader, test_loader = get_cifar10_loaders(config)
    
    return train_loader, test_loader
