import re
import os
from torch import Tensor
from torch.utils.data import IterableDataset
from torchvision import transforms
from PIL import Image

# from torchvision.io import read_image


# image dimensions to enforce for all images
SAMPLE_SIZE = (300, 400)


def get_label(path):
    """Get the label from the name of a sample's filepath."""
    r = re.compile("([a-z]+)([0-9])+")
    m = r.match(path)
    label = m.group(1)
    if label == "covid":
        return Tensor([[0]])
    elif label == "healthy":
        return Tensor([[1]])
    elif label == "other":
        return Tensor([[2]])


# Data normalization
im_xform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),  # Convert image to grayscale
        transforms.ToTensor(),
        # NOTE: normalized to mean 0 and variance 1
        transforms.Normalize([0], [1]),
        transforms.Resize(SAMPLE_SIZE),
    ]
)


class ImageDataSet(IterableDataset):
    def __init__(self, im_paths):

        self.im_paths = im_paths
        self.samples = []

        # read images
        for path in im_paths:
            label = get_label(os.path.split(path)[1])
            image = im_xform(Image.open(path))
            # image = im_xform(read_image(path))
            self.samples.append((image, label))

    def __len__(self):
        """Get the number of images in this dataset."""
        return len(self.im_paths)

    def __iter__(self):
        # Here we have to return the item requested by `idx`.
        # The PyTorch DataLoader class will use this method to make an iterable for training/validation loop.
        return iter(self.samples)
