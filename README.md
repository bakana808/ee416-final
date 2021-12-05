# EE416 Final Project
---

A convolutional neural network that classifies a chest CT image of various patients as having COVID-19 pneumonia, other pneumonia, or being healthy.

## Overview

### Dataset
-   ~4,400 2D lung CT images
-   3 classes/labels
    -   COVID-19 pneumonia
    -   other pneumonia
    -   healthy condition

### Test & Training Data
-   split 80% training and 20% test using 5-fold cross validation
-   (optional) data augmentation by rotating images 90, 180, and 270 degrees to quadruple the amount of images

### Tasks
1.  create a program that splits the dataset into training and test data and automatically forms a folder structure
    -   80% training, 20% test using 5-fold cross validation
2.  finish the convolutional neural network (CNN) by implementing the 2nd layer convolution kernels, convlution operator, and summation
    -   determine amount of parameters in this CNN
3.  train the model using the training dataset
    -   decide model architecture
    -   decide training loss function
    -   decide optimizer
    -   decide training parameters
4.  test the model on the test dataset
    -   outputs an accuracy
    -   **target: >85% (partial credit: >75%)**
5.  create report, which includes:
    -   detail of CNN model and hyperparameters used
    -   graphs of training and test loss across epochs (iterations)
    -   accuracy of best model on the test set

### Notes:
FCN: fully-connected network
CNN: convolutional neural network

## Running
Before running the program, place the folder of images labeled "COVID" in `Data/Covid`, the folder of images labeled "Healthy" in `Data/Healthy`, and the folder of images labeled "Other pnumonia" in `Data/Others`. The program will automatically move and rename these files.

Then, you can run:
`python eg.py`