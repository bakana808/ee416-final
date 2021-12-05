import torch
from torch import nn
from torchvision import models

# MODEL PARAMTERS
# ===============

# number of samples to process at once
BATCH_SIZE = 100

# if true, linear layers will learn an additive bias
NN_LINEAR_BIAS = True

# NOTE: each image contains roughly ~90,000 pixels = ~90,000 features

# number of features to output in the first linear layer
NN_LINEAR_1_FOUT = 1000

# number of features to input in the second linear layer
NN_LINEAR_2_FIN = 1000


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: [Transfer learning with pre-trained ResNet-50] Design your own fully-connected network (FCN) classifier.
        # Design your own FCN classifier. Here I provide a sample of two-layer FCN.
        # Refer to PyTorch documentations of torch.nn to pick your layers. (https://pytorch.org/docs/stable/nn.html)
        # Some common Choices are: Linear, ReLU, Dropout, MaxPool2d, AvgPool2d
        # If you have many layers, consider using nn.Sequential() to simplify your code

        # Load pretrained ResNet-50
        self.model_resnet = models.resnet50(pretrained=True)

        # Set ResNet-50's FCN as an identity mapping
        num_fc_in = self.model_resnet.fc.in_features
        self.model_resnet.fc = nn.Identity()

        # TODO: Design your own FCN
        self.fc1 = nn.Linear(
            num_fc_in, NN_LINEAR_1_FOUT, bias=NN_LINEAR_BIAS
        )  # from input of size num_fc_in to output of size ?
        self.fc2 = nn.Linear(
            NN_LINEAR_2_FIN, 3, bias=NN_LINEAR_BIAS
        )  # from hidden layer to 3 class scores
        self.flt1 = nn.Flatten(0, -2)
        self.flt2 = nn.Flatten()
        self.dense = nn.Linear(3, 3)

    def forward(self, x):
        # TODO: Design your own network, implement forward pass here

        # No need to define self.relu because it contains no parameters
        relu = nn.ReLU()

        with torch.no_grad():
            features = self.model_resnet(x)

        # Activation are flattened before being passed to the fully connected layers
        x = self.fc1(features)
        x = relu(x)
        x = self.fc2(x)
        x = torch.flatten(x, 1)
        # x = self.dense(x)

        # The loss layer will be applied outside Network class
        return x
