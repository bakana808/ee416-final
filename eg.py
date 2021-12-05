#%%
import re

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

from dataset import DataSet
from cnn import Network

# ================================================================================
# PROGRAM PARAMETERS
# ================================================================================

# path to covid images
DATASET_COVID_PATH = "Data/Covid"

# path to healthy images
DATASET_HEALTHY_PATH = "Data/Healthy"

# path to "other pnumonia" images
DATASET_OTHER_PATH = "Data/Others"

# path to output all images to
DATASET_OUTPUT = "Dataset"

# ================================================================================
# TRAINING PARAMETERS
# ================================================================================

# number of times to train the model on the same dataset
# more epochs = longer processing
NUM_EPOCHS = 100

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


def get_label(path):
    """Get the label from the name of a sample's filepath."""

    r = re.compile("([a-z]+)([0-9])+")
    m = r.match(path)
    label = m.group(1)
    return label


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

print(f"Moved a total of {moved_samples} files.")

# %% DATASET SPLITTING
# ================================================================================

# list of samples (images)
samples = os.listdir(DATASET_OUTPUT)
print(f"Total samples = {len(samples)}")

random.shuffle(samples)  # randomize order of samples

folds = np.array_split(samples, 5)

# %% DATASET PREPROCESSING
# ================================================================================

# Data path
DATA_train_path = Dataset("./Dataset/Train")
DATA_test_path = Dataset("./Dataset/Test")

# Data normalization
MyTransform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
        transforms.ToTensor(),  # Transform
        # TODO: Normalize to zero mean and unit variance with appropriate parameters
        # from [0,255] uint8 to [0,1] float
        transforms.Normalize([0], [1]),
    ]
)

DATA_train = datasets.ImageFolder(root=DATA_train_path, transform=MyTransform)
DATA_test = datasets.ImageFolder(root=DATA_test_path, transform=MyTransform)

print("Done!")

# Create dataloaders
# TODO: Experiment with different batch sizes
trainloader = DataLoader(Data_train, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(Data_test, batch_size=BATCH_SIZE, shuffle=True)

## declaration of Network

# %% MODEL SETUP
# ================================================================================

# the hardware device (gpu or cpu) to use when training
device = "cuda" if torch.cuda.is_available() else "cpu"

# our CNN model
model = Network().to(device)

# our loss function
criterion = (
    nn.CrossEntropyLoss()
)  # Specify the loss layer (note: CrossEntropyLoss already includes LogSoftMax())
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
            pred = model(batch)  # This will call Network.forward() that you implement
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

train(model, trainloader, num_epochs)
print("Evaluate on test set")
evaluate(model, testloader)
