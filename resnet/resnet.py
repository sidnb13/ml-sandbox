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
from typing import *

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
    def __init__(
        self, in_channels, pre_expansion_channels, stride=1, use_batchnorm=True, expansion=1
    ) -> None:
        super().__init__()
        self.expansion = expansion

        self.conv1 = nn.Conv2d(
            in_channels, pre_expansion_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(pre_expansion_channels) if use_batchnorm else nn.Identity()

        self.conv2 = nn.Conv2d(
            in_channels, pre_expansion_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(pre_expansion_channels) if use_batchnorm else nn.Identity()

        self.relu = nn.ReLU()

        if in_channels != pre_expansion_channels * expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(pre_expansion_channels, pre_expansion_channels * expansion, kernel_size=1, stride=stride),
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
        self,
        in_channels,
        pre_expansion_channels,
        stride=1,
        use_batchnorm=True,
        expansion=4,
    ) -> None:
        super().__init__()
        self.expansion = expansion

        # No batchnorm applied to the first conv layer
        self.conv1 = (
            nn.Conv2d(
                in_channels, pre_expansion_channels, kernel_size=1, stride=1, padding=1
            )
            if in_channels != pre_expansion_channels
            else nn.Identity()
        )

        self.conv2 = nn.Conv2d(
            pre_expansion_channels,
            pre_expansion_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.bn2 = (
            nn.BatchNorm2d(pre_expansion_channels) if use_batchnorm else nn.Identity()
        )

        self.conv3 = nn.Conv2d(
            pre_expansion_channels, pre_expansion_channels * expansion
        )
        self.bn3 = (
            nn.BatchNorm2d(pre_expansion_channels * expansion)
            if use_batchnorm
            else nn.Identity()
        )

        self.relu = nn.ReLU()

        # skip connection follows conv1
        self.shortcut = nn.Conv2d(
            in_channels,
            pre_expansion_channels * expansion,
            kernel_size=1,
            stride=stride,
        )

    def forward(self, x: torch.Tensor):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out_skip)
        out = self.relu(out)
        out = self.conv3(out)

        out += self.shortcut(x)
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, input_channels=64, use_batchnorm=True) -> None:
        super().__init__()

        # Pre-residual conv layer
        self.conv1 = nn.Conv2d(3, input_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(input_channels) if use_batchnorm else nn.Identity()

        # Residual blocks
        self.layer1 = self._create_layer(ResidualBlock, 64, num_blks=2, stride=1)
        self.layer2 = self._create_layer(Bottleneck, 128, num_blks=2, stride=2)
        self.layer3 = self._create_layer(Bottleneck, 256, num_blks=2, stride=2)
        self.layer4 = self._create_layer(Bottleneck, 512, num_blks=2, stride=2)

        # Linear output
        self.readout = nn.Sequential(
            [
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, config.num_classes),
            ]
        )

    def _create_layer(
        self,
        block: Type[ResidualBlock | Bottleneck],
        channels: int,
        num_blks=1,
        stride=1,
    ):
        # Assign correct strides to preserve channels
        stride_pattern = [stride] + [1] * (num_blks - 1)
        layers = []
        
        in_channels = self.in_channels
        
        for stride in stride_pattern:
            layers.append(block(in_channels, channels, stride=stride))
            in_channels = channels * block.expansion

    def forward(self, x: torch.Tensor):
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
