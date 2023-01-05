"""Implementation of a variational autoencoder (VAE) in PyTorch on MNIST data."""

import config
import matplotlib.pyplot as plt
import numpy as np
import torch
from absl import app, flags, logging
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        pass


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        pass


class VAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        pass

def kl_divergence(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var)

def main(argv):
    pass


if __name__ == "__main__":
    app.run(main)
