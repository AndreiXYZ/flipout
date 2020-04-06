import torch
import numpy as np
import PIL
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def mnist_dataloaders(config):
    transformations = [transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize(mean=[0.1307], std=[0.3081])]
    
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose(transformations))
    
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose(transformations))

    train_loader = DataLoader(train_set, 
                              batch_size = config['batch_size'],
                              shuffle = True, 
                              pin_memory = True,
                              num_workers = 8,
                              drop_last = False)
    test_loader = DataLoader(test_set,
                             batch_size = config['test_batch_size'],
                             shuffle=False,
                             pin_memory = True,
                             num_workers = 8,
                             drop_last = False)
    
    
    return train_loader, test_loader


def cifar10_dataloaders(config):
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
                             batch_size = config['test_batch_size'],
                             shuffle = False,
                             pin_memory = True,
                             num_workers = 8,
                             drop_last = False)
    

    return train_loader, test_loader


def image_loader(path):
    img = Image.open(path)
    return img

def is_valid_file(path):
    extension = path.split('.')[-1]
    if extension == 'JPEG':
        return True
    return False

def imagenette_dataloaders(config):
    transforms_train = transforms.Compose([transforms.Resize((224,224)),
                                           transforms.RandomCrop(224, padding=28),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor()])

    transforms_test = transforms.Compose([transforms.Resize((224,224)),
                                          transforms.ToTensor()])

    train_set = datasets.DatasetFolder(root='./data/imagenette2/train', loader=image_loader,
                                    is_valid_file=is_valid_file, transform=transforms_train)

    test_set = datasets.DatasetFolder(root='./data/imagenette2/val', loader=image_loader,
                                    is_valid_file=is_valid_file, transform=transforms_test)

    train_loader = DataLoader(train_set, 
                                batch_size = config['batch_size'],
                                shuffle = True, 
                                pin_memory = True,
                                num_workers = 8,
                                drop_last = False)

    test_loader = DataLoader(test_set,
                                batch_size = config['test_batch_size'],
                                shuffle=False,
                                pin_memory = True,
                                num_workers = 8,
                                drop_last = False)

    return train_loader, test_loader