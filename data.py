# -*- coding: utf-8 -*-

__author__ = "Rahul Bhalley"

from config import *

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dsets


def load_data():

    # Preprocessing
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(LOAD_DIM),
        transforms.RandomCrop(CROP_DIM),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    # Make datasets
    X_folder = dsets.ImageFolder(DATASET_PATH["trainA"], transform=transform)
    Y_folder = dsets.ImageFolder(DATASET_PATH["trainB"], transform=transform)
    
    # Make dataset loaders
    X_set = DataLoader(X_folder, batch_size=BATCH_SIZE, shuffle=True)
    Y_set = DataLoader(Y_folder, batch_size=BATCH_SIZE, shuffle=True)
    
    # Print length of sample batches
    print("Dataset Details")
    print(f"X_set batches: {len(X_set)}")
    print(f"Y_set batches: {len(Y_set)}")
    print("")

    # Return the datasets
    return X_set, Y_set

def get_infinite_X_data(X_set):
    while True:
        for x, _ in X_set:
            yield x

def get_infinite_Y_data(Y_set):
    while True:
        for y, _ in Y_set:
            yield y

# There's some problem with batch size 
# of sampled data using `torchvision`.
# 
# This block of code tries to 
# eliminate the problem.

def safe_sampling(X_data, Y_data, device):

    # First sample the data
    x_sample, y_sample = next(X_data), next(Y_data)
    
    # Check requirement conditions
    # and sample next accordingly.
    if x_sample.size(0) != BATCH_SIZE:  # condition for `x_sample`
        print(f"Batch size not equal to that of x_sample: {BATCH_SIZE} != {x_sample.size(0)} | skipping...")
        x_sample = next(X_data)
    if y_sample.size(0) != BATCH_SIZE:  # condition for `y_sample`
        print(f"Batch size not equal to that of y_sample: {BATCH_SIZE} != {y_sample.size(0)} | skipping...")
        y_sample = next(Y_data)
    
    # Return correct data
    return x_sample.to(device), y_sample.to(device)