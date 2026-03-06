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
    A_folder = dsets.ImageFolder(config.DATASET_PATH["trainA"], transform=transform)
    B_folder = dsets.ImageFolder(config.DATASET_PATH["trainB"], transform=transform)
    
    # Make dataset loaders with drop_last=True to handle uneven batch sizes
    loader_a = DataLoader(
        A_folder, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        drop_last=True,
        num_workers=2,
        pin_memory=True
    )
    loader_b = DataLoader(
        B_folder, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        drop_last=True,
        num_workers=2,
        pin_memory=True
    )
    
    # Print length of sample batches
    print("Dataset Details")
    print(f"Domain A batches: {len(loader_a)}")
    print(f"Domain B batches: {len(loader_b)}")
    print("")

    # Return infinite iterators
    return itertools.cycle(loader_a), itertools.cycle(loader_b)

def safe_sampling(iter_a, iter_b, device):
    # Sample the data
    real_a, _ = next(iter_a)
    real_b, _ = next(iter_b)
    
    # Return correct data with channels_last if possible (handled in main usually, but can be done here)
    return real_a.to(device, non_blocking=True), real_b.to(device, non_blocking=True)
