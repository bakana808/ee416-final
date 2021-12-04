#%%

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
# TRAINING PARAMETERS
# ================================================================================

# number of times to train the model on the same dataset
# more epochs = longer processing
NUM_EPOCHS = 100

# ================================================================================
# OPTIMIZER PARAMETERS
# ================================================================================

# learning rate
LR = 0.001

# epsilon; the term added to the denominator to improve numerical stability
EPS = 1e-8

# penalty
WEIGHT_DECAY = 0


# ================================================================================
# DATASET ORGANIZATION & SPLITTING
# ================================================================================

# Load the dataset and train and test splits
print("Loading datasets...")

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

# MODEL SETUP
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
