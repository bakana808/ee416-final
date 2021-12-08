from torch.utils.data import Dataset
from torchvision import transforms

sample_xform = transforms.Compose([transforms.ToTensor()])


class ImageDataSet(Dataset):
    def __init__(self, samples):

        self.samples = samples

    def __len__(self):
        """Get the number of images in this dataset."""
        return len(self.samples)

    def __getitem__(self, index):
        """Convert this dataset into an iterable."""
        return (
            self.samples[index][0],
            self.samples[index][1],
        )
