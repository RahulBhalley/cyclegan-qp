# -*- coding: utf-8 -*-

from config import config
import itertools
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dsets

def load_data():
    # Preprocessing
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(config.LOAD_DIM),
        transforms.RandomCrop(config.CROP_DIM),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    # Make datasets
    X_folder = dsets.ImageFolder(config.DATASET_PATH["trainA"], transform=transform)
    Y_folder = dsets.ImageFolder(config.DATASET_PATH["trainB"], transform=transform)
    
    # Make dataset loaders with drop_last=True to handle uneven batch sizes
    X_loader = DataLoader(
        X_folder, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        drop_last=True,
        num_workers=2,
        pin_memory=True
    )
    Y_loader = DataLoader(
        Y_folder, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        drop_last=True,
        num_workers=2,
        pin_memory=True
    )
    
    # Print length of sample batches
    print("Dataset Details")
    print(f"X_set batches: {len(X_loader)}")
    print(f"Y_set batches: {len(Y_loader)}")
    print("")

    # Return infinite iterators
    return itertools.cycle(X_loader), itertools.cycle(Y_loader)

def safe_sampling(X_iter, Y_iter, device):
    # Sample the data
    x_sample, _ = next(X_iter)
    y_sample, _ = next(Y_iter)
    
    # Return correct data with channels_last if possible (handled in main usually, but can be done here)
    return x_sample.to(device, non_blocking=True), y_sample.to(device, non_blocking=True)
