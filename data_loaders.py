import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_mnist_loaders(config):
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                                                        normalize]))
                                                                                                        
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                                                        normalize]))

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
    transform_train = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

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
