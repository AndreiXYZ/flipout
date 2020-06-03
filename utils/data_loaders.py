import torch
import math
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def mnist_dataloaders(config):
    transformations = transforms.Compose([
                                          transforms.Resize((32,32)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307), (0.3081))
                                         ])
    
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
    val_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Grab train and test sets if not using validation
    if not config['val']:
        train_size = len(train_set)
        val_size = 0
        test_size = len(test_set)

        train_loader = DataLoader(train_set,
                                batch_size = config['batch_size'],
                                shuffle = True,
                                pin_memory = True,
                                num_workers = 8,
                                drop_last = False)
        
        val_loader = None
        test_loader = DataLoader(test_set,
                    batch_size = config['test_batch_size'],
                    shuffle = False,
                    pin_memory = True,
                    num_workers = 8,
                    drop_last = False)
    
    else:
        idxs = list(range(len(train_set)))
        train_idxs = len(train_set) - config['val_size']
        train_sampler = SubsetRandomSampler(idxs[:train_idxs])
        val_sampler = SubsetRandomSampler(idxs[train_idxs:])

        # Determine sizes
        train_size = len(train_set) - config['val_size']
        val_size = config['val_size']
        test_size = len(test_set)

        train_loader = DataLoader(train_set,
                        batch_size = config['batch_size'],
                        sampler=train_sampler,
                        pin_memory = True,
                        num_workers = 8,
                        drop_last = False)
        
        val_loader = DataLoader(val_set,
                        batch_size=config['test_batch_size'],
                        sampler=val_sampler,
                        pin_memory=True,
                        num_workers= 8,
                        drop_last=False)
        
        test_loader = DataLoader(test_set,
            batch_size = config['test_batch_size'],
            shuffle = False,
            pin_memory = True,
            num_workers = 8,
            drop_last = False)

    return (train_loader, val_loader, test_loader), (train_size, val_size, test_size)

def image_loader(path):
    img = Image.open(path)
    # Convert image to rgb if it's grayscale
    if img.mode!='RGB':
        arr = np.array(img)
        new_arr = arr[:, :, np.newaxis]
        new_arr = np.repeat(new_arr, repeats=3, axis=2)
        img = Image.fromarray(new_arr)
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
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.4625, 0.4580, 0.4295),(0.3901, 0.3880, 0.4042))
                                           ])

    transforms_test = transforms.Compose([transforms.Resize((224,224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4625, 0.4580, 0.4295),(0.3901, 0.3880, 0.4042))
                                          ])

    train_set = datasets.DatasetFolder(root='./data/imagenette2/train', loader=image_loader,
                                    is_valid_file=is_valid_file, transform=transforms_train)

    val_set = datasets.DatasetFolder(root='./data/imagenette2/train', loader=image_loader,
                                    is_valid_file=is_valid_file, transform=transforms_test)
    
    test_set = datasets.DatasetFolder(root='./data/imagenette2/val', loader=image_loader,
                                    is_valid_file=is_valid_file, transform=transforms_test)

    if not config['val']:
        train_size = len(train_set)
        val_size = 0
        test_size = len(test_set)

        train_loader = DataLoader(train_set, 
                                    batch_size = config['batch_size'],
                                    shuffle = True, 
                                    pin_memory = True,
                                    num_workers = 8,
                                    drop_last = False)

        val_loader = None
        
        test_loader = DataLoader(test_set,
                                    batch_size = config['test_batch_size'],
                                    shuffle=False,
                                    pin_memory = True,
                                    num_workers = 8,
                                    drop_last = False)

    else:
        idxs = list(range(len(train_set)))
        idxs = np.array(idxs)
        np.random.shuffle(idxs)
        train_idxs = len(train_set) - config['val_size']
        train_sampler = SubsetRandomSampler(idxs[:train_idxs])
        val_sampler = SubsetRandomSampler(idxs[train_idxs:])

        # Determine sizes
        train_size = len(train_set) - config['val_size']
        val_size = config['val_size']
        test_size = len(test_set)
        train_loader = DataLoader(train_set,
                        batch_size = config['batch_size'],
                        sampler=train_sampler,
                        pin_memory = True,
                        num_workers = 8,
                        drop_last = False)
        
        val_loader = DataLoader(val_set,
                        batch_size=config['test_batch_size'],
                        sampler=val_sampler,
                        pin_memory=True,
                        num_workers= 8,
                        drop_last=False)

        test_loader = DataLoader(test_set,
                        batch_size = config['test_batch_size'],
                        shuffle = False,
                        pin_memory = True,
                        num_workers = 8,
                        drop_last = False)
        
    return (train_loader, val_loader, test_loader), (train_size, val_size, test_size)