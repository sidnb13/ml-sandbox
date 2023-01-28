from datetime import datetime

import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from absl import app, flags, logging
from colorama import Fore, Style
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLAGS = flags.FLAGS
flags.DEFINE_boolean("use_wandb", True, "Use wandb for experiment logging.")

timestamp = torch.tensor(datetime.now().timestamp()).to(device)
timestamp = datetime.fromtimestamp(timestamp.int()).strftime("%Y-%m-%d-%H-%M-%S")


class Generator(nn.Module):
    def __init__(self, latent_dim: int, image_dim: int, act: str = "ReLU") -> None:
        super().__init__()
        self.act = act
        self.latent_dim = latent_dim
        self.image_dim = image_dim

        self.lin_list = torch.nn.ModuleList()
        prev = self.latent_dim

        for hidden_dim in config.hidden_dims:
            self.lin_list.append(create_block(prev, hidden_dim, self.act))
            prev = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin_list(x)


class Discriminator(nn.Module):
    def __init__(self, in_dim: int, act: str = "ReLU") -> None:
        super().__init__()
        self.act = act
        self.in_dim = in_dim

        self.lin_list = nn.ModuleList()
        prev = self.in_dim

        for hidden_dim in config.hidden_dims[::-1]:
            self.lin_list.append(create_block(prev, hidden_dim, self.act))
            prev = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin_list(x)


def create_block(in_dim: int, out_dim: int, act) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        getattr(nn, act)(),
    )


def discriminator_loss(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    pass


def generator_loss(self, fake: torch.Tensor) -> torch.Tensor:
    pass


def sample_generator(self, n: int) -> torch.Tensor:
    pass


def main(argv):
    del argv

    if FLAGS.use_wandb:
        wandb.init(
            settings=wandb.Settings(start_method="fork"),
            project=config.wandb_config.get("wandb_project", "matdeeplearn"),
            entity=config.wandb_config.get("wandb_entity", "fung-lab"),
            name=config.wandb_config.get("wandb_name", "gan") + f"-{timestamp}",
            config=config.wandb_config,
        )

    # prepare dataset
    train_loader = DataLoader(
        datasets.MNIST(
            config.data_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        ),
        batch_size=config.batch_size,
    )
    test_loader = DataLoader(
        datasets.MNIST(
            config.data_dir,
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        ),
        batch_size=config.batch_size,
    )

    # normalization
    mean_tr = mean_ts = 0.0
    std_tr = std_ts = 0.0

    for (data_tr, _) in train_loader:
        mean_tr += np.mean(data_tr.numpy(), axis=(0, 2, 3))
        std_tr += np.mean(np.square(data_tr.numpy()), axis=(0, 2, 3))

    for (data_ts, _) in test_loader:
        mean_ts += np.mean(data_ts.numpy(), axis=(0, 2, 3))
        std_ts += np.mean(np.square(data_ts.numpy()), axis=(0, 2, 3))

    mean_tr /= len(train_loader)
    std_tr /= len(train_loader)
    mean_ts = np.sqrt(np.mean(std_ts))
    std_ts = np.sqrt(np.mean(std_ts))

    # initialize models and optimizer
    generator = Generator(config.latent_dim, np.prod(config.image_size)).to(device)
    discriminator = Discriminator(np.prod(config.image_size)).to(device)

    # initialize optimizers
    opt_gen = optim.SGD(
        generator.parameters(),
        lr=config.lr,
        momentum=config.init_momentum,
    )
    opt_disc = optim.SGD(
        discriminator.parameters(),
        lr=config.lr,
        momentum=config.init_momentum,
    )

    # train model
    for step in range(config.steps):
        # train discriminator for k steps
        # train generator
        # evaluate on test set
        # log metrics
        pass


if __name__ == "__main__":
    app.run(main)
