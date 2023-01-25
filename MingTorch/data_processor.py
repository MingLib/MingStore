import os, sys

import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler

def data_split(dataset, batch_size, validation_split=.2, shuffle=False, random_seed=None, num_workers=20, pin_memory=True):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(validation_split * dataset_size)
    if shuffle:
        if random_seed is not None:
            np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, validation_loader