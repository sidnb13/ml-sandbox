"""
Fashion MNIST dataset convenience wrapper.
"""

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST


class FashionMNISTSet:
    def __init__(self, dim=(28, 28), batch_size=32) -> None:
        composition = transforms.Compose(
            [transforms.Resize(dim), transforms.ToTensor()]
        )
        self.train_set = FashionMNIST(
            root="./dataset", train=True, download=True, transform=composition
        )
        self.val_set = FashionMNIST(
            root="./dataset", train=False, download=True, transform=composition
        )
        self.batch_size = batch_size
        self.dim = dim

    def create_dataloader(self, train: bool):
        """Create a dataloader."""
        if train:
            shuffle = True
            dataset = self.train_set
        else:
            shuffle = False
            dataset = self.val_set
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle
        )

    def text_labels(self, indices):
        """Return text labels."""
        labels = [
            "t-shirt",
            "trouser",
            "pullover",
            "dress",
            "coat",
            "sandal",
            "shirt",
            "sneaker",
            "bag",
            "ankle boot",
        ]
        return [labels[int(i)] for i in indices]

    def visualize(self, batch, nrows=1, ncols=8, labels=[]):
        X, y = batch
        if not labels:
            labels = self.text_labels(y)

        _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 8))

        for i, ax in enumerate(axes.flat):
            ax.imshow(X[i].squeeze(0), cmap="gray")
            ax.set_title(labels[i])
            ax.axis("off")
