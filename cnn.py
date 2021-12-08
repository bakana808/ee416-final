import torch
from torch import nn
from torchvision import models

from config import *


class Network_V2(nn.Module):
    def __init__(self):
        super().__init__()

        # NOTE: working with 300x400x1 images

        # 3x3 kernel, 1 -> 3 channels
        self.conv1 = nn.Conv2d(1, 3, 3, padding="same")

        # 5x5 kernel, 3 -> 6 channels
        self.conv2 = nn.Conv2d(3, 6, 5, padding="same")

        # 3x3 kernel, 6 -> 12 channels
        self.conv3 = nn.Conv2d(6, 12, 3, padding="same")

        # 5x5 kernel, 12 -> 25 channels
        self.conv4 = nn.Conv2d(12, 25, 5, padding="same")

        # ReLU activation
        self.relu = nn.ReLU()

        self.softmax = nn.Softmax()

        self.pool = nn.MaxPool2d(2, 2)

        # flatten (lower dimensions)
        self.flat1 = nn.Flatten(0, -1)

        self.lin1 = nn.Linear(15625, 256)

        self.lin2 = nn.Linear(256, 128)

        self.lin3 = nn.Linear(128, 3)

    def forward(self, x):

        # 1st 2D conv
        x = self.pool(self.relu(self.conv1(x)))

        # 2nd 2D conv
        x = self.pool(self.relu(self.conv2(x)))

        # 3rd 2D conv
        x = self.pool(self.relu(self.conv3(x)))

        # 4th 2D conv
        x = self.pool(self.relu(self.conv4(x)))

        # flatten
        # x = self.flat1(x)
        # print("shape = %s" % [x.shape])
        x = torch.flatten(x, 1, 3)
        # print("shape = %s" % [x.shape])

        x = self.relu(self.lin1(x))

        x = self.relu(self.lin2(x))

        # reduce to 3 features
        x = self.lin3(x)

        return x


class Network_V1(nn.Module):
    """Original NN design (from skeleton code).

    Around 75% accurate on the test set (20% of the data) when trained on 80% of the data.

    NOTE: This model expects 3-channel images. Will not accept 1-channel.
    """

    def __init__(self):
        super().__init__()
        # TODO: [Transfer learning with pre-trained ResNet-50] Design your own fully-connected network (FCN) classifier.
        # Design your own FCN classifier. Here I provide a sample of two-layer FCN.
        # Refer to PyTorch documentations of torch.nn to pick your layers. (https://pytorch.org/docs/stable/nn.html)
        # Some common Choices are: Linear, ReLU, Dropout, MaxPool2d, AvgPool2d
        # If you have many layers, consider using nn.Sequential() to simplify your code

        # NOTE:

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

        # The loss layer will be applied outside Network class
        return x
