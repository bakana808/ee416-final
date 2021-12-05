#%%
import re
import importlib

import os
import random

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split

from dataset import ImageDataSet
import cnn

# ================================================================================
# PROGRAM PARAMETERS
# ================================================================================

CLASSES = ("covid", "healthy", "other")

# path to covid images
DATASET_COVID_PATH = "Data/Covid"

# path to healthy images
DATASET_HEALTHY_PATH = "Data/Healthy"

# path to "other pnumonia" images
DATASET_OTHER_PATH = "Data/Others"

# path to output all images to
DATASET_OUTPUT = "Dataset"

# path to move all images to
DATASET_TRAIN = "Train"

# ================================================================================
# TRAINING PARAMETERS
# ================================================================================

# number of samples to process through the model at a time
BATCH_SIZE = 100

# number of times to train the model on the same dataset
# more epochs = longer processing
NUM_EPOCHS = 2

# ================================================================================
# OPTIMIZER PARAMETERS
# ================================================================================

# learning rate; the higher this number is, the faster the weights adjust when training
LR = 0.001

# epsilon; the term added to the denominator to improve numerical stability
EPS = 1e-8

# penalty
WEIGHT_DECAY = 0

# %% DATASET ORGANIZATION
# ================================================================================


def consolidate_data(dir, new_dir, label, req_ext=".png"):
    """Move all the files (recursively) in dir to new_dir.

    The files will be prefixed with `label` and suffixed
    with an incrementing number starting at 0.

    Returns the number of files moved.
    """
    print(f"Moving files from {dir} to {new_dir} (labeled '{label}')...")

    num_files = 0
    for root, _, files in os.walk(dir):
        for filename in files:
            ext = os.path.splitext(filename)[1]  # file extension
            path = root + os.sep + filename
            new_path = f"{new_dir}\\{label}{num_files}{ext}"
            if ext == req_ext:
                os.replace(path, new_path)
                num_files += 1

    print(f"Moved {num_files} files.")
    return num_files


# perform dataset consolidation
moved_samples = 0
moved_samples += consolidate_data(DATASET_COVID_PATH, DATASET_OUTPUT, "covid")
moved_samples += consolidate_data(DATASET_HEALTHY_PATH, DATASET_OUTPUT, "healthy")
moved_samples += consolidate_data(DATASET_OTHER_PATH, DATASET_OUTPUT, "other")

print(f"Moved a total of {moved_samples} files")

# %%
# DATASET SPLITTING
# ================================================================================

# list of samples (images)
samples = [DATASET_OUTPUT + "/" + p for p in os.listdir(DATASET_OUTPUT)]
print(f"Total samples = {len(samples)}")

# randomize order of samples
random.shuffle(samples)

k = 5  # k to use in k-fold cross-validation
folds = np.array_split(samples, k)
print(f"Split samples with k = {k}")
for i, fold in enumerate(folds):
    print(f"\tGroup {i}: {len(fold)} samples")

# %%
# DATASET PREPROCESSING
# ================================================================================


def split_dataset(n=0):
    """Split the dataset into the training and test set.

    Given a k-fold split, n is a number between 0 and k-1 referencing
    which group in the fold to use as the test set.
    The remaining groups will be used as the training set.
    """
    test_set = folds[n]
    training_set = []
    for i in range(len(folds)):
        if i != n:
            training_set += list(folds[i])

    return training_set, test_set


# create splits
n = 0
print(f"Creating test/split with n = {n}")
paths_train, paths_test = split_dataset(n)

# create datasets
ds_train = ImageDataSet(paths_train)
ds_test = ImageDataSet(paths_test)

# imfolder_train = datasets.ImageFolder(root=DATA_train_path, transform=MyTransform)
# imfolder_test = datasets.ImageFolder(root=DATA_test_path, transform=MyTransform)

# create dataloaders
ld_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=False)
ld_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

## declaration of Network

# %%
# MODEL SETUP
# ================================================================================

importlib.reload(cnn)  # reimports the network

# the hardware device (gpu or cpu) to use when training
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: %s" % device)

# our CNN model
model = cnn.Network().to(device)

# our loss function
criterion = nn.CrossEntropyLoss()
# Specify the loss layer (note: CrossEntropyLoss already includes LogSoftMax())
# TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)

# our optimizer that will be used when training
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)  # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength (default: lr=1e-2, weight_decay=1e-4)


def train(model, loader, num_epoch=NUM_EPOCHS):  # Train the model
    print("Start training...")
    model.train()  # Set the model to training mode
    for i in range(num_epoch):
        running_loss = []
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)

            optimizer.zero_grad()  # Clear gradients from the previous iteration

            label = torch.flatten(label).long()
            pred = model(batch)

            loss = criterion(pred, label)  # Calculate the loss

            running_loss.append(loss.item())
            loss.backward()  # Backprop gradients to all tensors in the network
            optimizer.step()  # Update trainable weights
        print(
            "Epoch {} loss:{}".format(i + 1, np.mean(running_loss))
        )  # Print the average loss for this epoch
    print("Done!")


def evaluate(model, loader):  # Evaluate accuracy on validation / test set
    model.eval()  # Set the model to evaluation mode
    correct = 0
    with torch.no_grad():  # Do not calculate grident to speed up computation
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            correct += (torch.argmax(pred, dim=1) == label).sum().item()
    acc = correct / len(loader.dataset)
    print("Evaluation accuracy: {}".format(acc))
    return acc


# MODEL TRAINING
# ================================================================================

train(model, ld_train, NUM_EPOCHS)
print("Evaluate on test set")
evaluate(model, ld_test)
