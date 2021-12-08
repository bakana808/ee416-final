#%%
import re
import importlib

import os
import random

from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import cross
from tqdm.autonotebook import tqdm, trange  # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split

from dataset import ImageDataSet
import cnn

from config import *
import config

# %%
# DATASET ORGANIZATION
# ================================================================================


def move_files(dir, new_dir, label, req_ext=".png"):
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
moved_samples += move_files(DATASET_COVID_PATH, DATASET_OUTPUT, "covid")
moved_samples += move_files(DATASET_HEALTHY_PATH, DATASET_OUTPUT, "healthy")
moved_samples += move_files(DATASET_OTHER_PATH, DATASET_OUTPUT, "other")

print(f"Moved a total of {moved_samples} files")

# %%
# DATASET PREPROCESSING
# ================================================================================

# list of samples paths (images)
sample_paths = [DATASET_OUTPUT + "/" + p for p in os.listdir(DATASET_OUTPUT)]
print(f"Total samples = {len(sample_paths)}")

# Data normalization
im_xform = transforms.Compose(
    [
        # NOTE: use 3 channels when using v1 network
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        # NOTE: normalized to mean 0 and variance 1
        transforms.Normalize([0], [1]),
        transforms.Resize(config.SAMPLE_SIZE),
    ]
)


def get_label(path):
    """Get the label from the name of a sample's filepath."""
    r = re.compile("([a-z]+)([0-9])+")
    m = r.match(path)
    label = m.group(1)
    if label == "covid":
        return torch.Tensor([[0]])
    elif label == "healthy":
        return torch.Tensor([[1]])
    elif label == "other":
        return torch.Tensor([[2]])


def read_images(im_paths):
    """Read all the images from the given paths.
    Each image will be transformed.

    A list of tuples (image, label) will be returned.
    """
    samples = []
    for path in tqdm(im_paths):
        label = get_label(os.path.split(path)[1])
        image = im_xform(Image.open(path))
        samples.append((image, label))
    return samples


print("Reading and transforming images...")
samples = read_images(sample_paths)


# %%
# DATASET SPLITTING
# ================================================================================

# # randomize order of samples
# random.shuffle(samples)

# k = 5  # k to use in k-fold cross-validation
# folds = np.array_split(samples, k)
# print(f"Splitting samples with k = {k}")
# for i, fold in enumerate(folds):
#     print(f"\tGroup {i}: {len(fold)} samples")


def split_dataset(folds, n=0):
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

    return np.array(training_set, dtype=object), np.array(test_set, dtype=object)


# create splits
# n = 0
# print(f"Creating test/training split with n = {n}")
# paths_train, paths_test = split_dataset(n)

# print("Creating datasets...")

# # create datasets
# ds_train = ImageDataSet(paths_train)
# ds_test = ImageDataSet(paths_test)

# # imfolder_train = datasets.ImageFolder(root=DATA_train_path, transform=MyTransform)
# # imfolder_test = datasets.ImageFolder(root=DATA_test_path, transform=MyTransform)

# print("Creating data loaders...")

# # create dataloaders
# ld_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=False)
# ld_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

## declaration of Network

# %%
# MODEL SETUP
# ================================================================================

importlib.reload(cnn)  # reimports the network
importlib.reload(config)

# the hardware device (gpu or cpu) to use when training
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: %s" % device)
if device == "cuda":
    torch.cuda.empty_cache()

# our CNN model
# model = cnn.Network_V1().to(device)
# model = cnn.Network_V2().to(device)

# our loss function


def train(model, loader, criterion, optimizer, num_epoch=1):  # Train the model
    model.train()  # Set the model to training mode
    with trange(num_epoch, desc="Epochs") as t:
        for i in t:
            # print("Epoch %d:" % (i + 1))
            running_loss = []
            for batch, label in tqdm(loader, desc="Batches", leave=False):
                batch = batch.to(device)
                label = label.to(device)

                optimizer.zero_grad()  # Clear gradients from the previous iteration

                label = torch.flatten(label).long()
                pred = model(batch)

                loss = criterion(pred, label)  # Calculate the loss

                running_loss.append(loss.item())
                loss.backward()  # Backprop gradients to all tensors in the network
                optimizer.step()  # Update trainable weights
            # Print the average loss for this epoch
            t.set_postfix(loss="%0.6f" % np.mean(running_loss))
    print("Finished training!")


def evaluate(model, loader):  # Evaluate accuracy on validation / test set
    model.eval()  # Set the model to evaluation mode
    correct = 0
    with torch.no_grad():  # Do not calculate grident to speed up computation
        for batch, label in tqdm(loader, desc="Batches"):
            batch = batch.to(device)
            label = label.to(device)
            label = torch.flatten(label).long()
            pred = model(batch)
            correct += (torch.argmax(pred, dim=1) == label).sum().item()
    acc = correct / len(loader.dataset)
    print("\taccuracy: %0.6f" % acc)
    return acc


# CROSS-VALIDATION
# ================================================================================


def plot_accuracy(accuracies):
    avg = np.mean(accuracies)

    plt.figure()
    plt.title("Cross-Validation Accuracy")
    plt.xlabel("Group")
    plt.ylabel("Accuracy")

    # bars
    bars = plt.bar(range(1, 6), accuracies, 0.5)
    bars[1].set_color("C1")
    bars[2].set_color("C2")
    bars[3].set_color("C3")
    bars[4].set_color("C4")

    # average line
    plt.axline([1, avg], [5, avg], dashes=[0, 2, 0], color="black")
    plt.text(5.5, avg, "%0.3f" % avg)


def cross_validate(k, model_type):

    # randomize order of samples
    random.shuffle(samples)

    folds = np.array_split(samples, k)
    print(f"Splitting samples with k = {k}")
    for i, fold in enumerate(folds):
        print(f"\tGroup {i}: {len(fold)} samples")

    accuracies = []

    # run training cycle for each group
    for n in range(k):

        # create model
        model = model_type().to(device)

        # set loss function
        # NOTE: CrossEntropyLoss already includes LogSoftMax()
        loss_fn = nn.CrossEntropyLoss()

        # set optimizer
        # NOTE: Specify optimizer and assign trainable parameters to it,
        # weight_decay is L2 regularization strength
        # (default: lr=1e-2, weight_decay=1e-4)
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR,
            weight_decay=config.WEIGHT_DECAY,
        )

        # create splits
        print(f"Creating test/training split, using group {n + 1} as test set...")
        paths_train, paths_test = split_dataset(folds, n)

        # create datasets
        ds_train = ImageDataSet(paths_train)
        ds_test = ImageDataSet(paths_test)

        # create dataloaders
        ld_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=False)
        ld_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

        # train model
        print("Training model for %d epochs..." % config.NUM_EPOCHS)
        train(model, ld_train, loss_fn, optimizer, config.NUM_EPOCHS)

        # evaluate model and save accuracy
        print("Evaluating model...")
        accuracies.append(evaluate(model, ld_test))

    # print accuracies
    print("Model estimated accuracies:")
    for n, acc in enumerate(accuracies):
        print("\tGroup %d: %0.6f" % (n + 1, accuracies[n]))
    print("\tAverage: %0.6f" % np.mean(accuracies))
    plot_accuracy(accuracies)


# perform 5-fold cross-validation
cross_validate(5, cnn.Network_V2)

# TRAIN
# ================================================================================
# train(model, ld_train, config.NUM_EPOCHS)
# print("Evaluate on test set")
# evaluate(model, ld_test)

# %%
