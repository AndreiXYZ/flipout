import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_mnist_loaders(config):
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.toTensor())
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.toTensor())

    train_loader = DataLoader(train_set, 
                              batch_size = config['batch_size'],
                              shuffle = True, 
                              pin_memory = True,
                              num_workers = 8)
    test_loader = DataLoader(test_set,
                             batch_size = config['batch_size'],
                             shuffle=True,
                             pin_memory = True,
                             num_workers = 8)
    
    return train_loader, test_loader


def get_cifar_loaders(config):
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.toTensor())
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.toTensor())

    train_loader = DataLoader(train_set,
                              batch_size = config['batch_size'],
                              shuffle = True,
                              pin_memory = True,
                              num_workers = 8)
    test_loader = DataLoader(test_set,
                             batch_size = config['batch_size'],
                             shuffle = True,
                             pin_memory = True,
                             num_workers = 8)
    
    return train_loader, test_loader