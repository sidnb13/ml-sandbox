"""ResNet implementation in PyTorch."""
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from absl import app
from absl import flags
from absl import logging
import config

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", "./data/mnist", "Directory to download data to.")
flags.DEFINE_float(
    "learning_rate", 0.001, "Learning rate for training.", short_name="lr"
)
flags.DEFINE_integer(
    "num_epochs", 10, "Number of epochs to train for.", short_name="ne"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        self.relu = nn.ReLU()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        # residual connection
        out += self.shortcut(x)

        return self.relu(out)


class Bottleneck(nn.Module):
    def __init__(
        self, in_channels, pre_expansion_channels, stride=1, expansion=4
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, pre_expansion_channels, kernel_size=1, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            pre_expansion_channels,
            pre_expansion_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            pre_expansion_channels, pre_expansion_channels * expansion
        )
        
        self.relu = nn.ReLU()
        
        # skip connection follows conv1
        self.shortcut = nn.Conv2d(in_channels, pre_expansion_channels * expansion, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor):
        out_skip = self.conv1(x)
        out_skip = self.relu(out_skip)
        
        # residual connection occurs after channel expansion
        out = self.conv2(out_skip)
        out = self.relu(out)
        out = self.conv3(out)
        
        out += self.shortcut(out_skip)
        return self.relu(out)

class ResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        pass


def train_step():
    pass


def train_loop():
    pass


def compute_metrics():
    pass


def see_data(dataloader: DataLoader, grid_dim=3):
    """Utility function, See some data samples from the dataloader"""
    # get some random training images
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    classes = config.classes

    # show images
    fig, axs = plt.subplots(grid_dim, grid_dim, figsize=(5, 5))
    for idx in np.arange(grid_dim**2):
        ax = axs[idx // grid_dim, idx % grid_dim]
        ax.imshow(images[idx].numpy().transpose(1, 2, 0))
        ax.set_title(classes[labels[idx].item()])

    plt.show()


def main(argv):
    # set up CIFAR10 toy dataset
    train_set = datasets.CIFAR10(
        FLAGS.data_dir, train=True, download=True, transform=transforms.ToTensor()
    )
    test_set = datasets.CIFAR10(
        FLAGS.data_dir, train=False, download=True, transform=transforms.ToTensor()
    )

    # set up data loader
    dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True, drop_last=False
    )


if __name__ == "__main__":
    app.run(main)
