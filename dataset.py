# TODO: Construct your data in the following baseline structure:
# 1) ./Dataset/Train/image/,
# 2) ./Dataset/Train/label,
# 3) ./Dataset/Test/image, and
# 4) ./Dataset/Test/label
class DataSet:
    def __init__(self, root):

        self.ROOT = root
        self.images = read_images(root + "/image")
        self.labels = read_labels(root + "/label")

    def __len__(self):
        # Return number of points in the dataset

        return len(self.images)

    def __getitem__(self, idx):
        # Here we have to return the item requested by `idx`. The PyTorch DataLoader class will use this method to make an iterable for training/validation loop.

        img = images[idx]
        label = labels[idx]

        return img, label
