import itertools
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


class AdversarialLoss:
    def __init__(self, gen: torch.nn.Module, disc: torch.nn.Module) -> None:
        self.gen = gen
        self.disc = disc

    def discriminator_loss(
        self, batch: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        return torch.log(self.disc(batch)) + torch.log(1 - self.disc(self.gen(noise)))

    def generator_loss(self, noise: torch.Tensor) -> torch.Tensor:
        return -torch.log(self.disc(self.gen(noise)))


def sample_generator(self, n: int) -> torch.Tensor:
    raise NotImplementedError("sample_generator not implemented")


def main(argv):
    del argv

    if FLAGS.use_wandb:
        wandb.init(
            settings=wandb.Settings(start_method="fork"),
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.job_id,
            config={
                "lr": config.lr,
                "batch_size": config.batch_size,
                "steps": config.steps,
                "dataset": "MNIST",
            },
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

    # initialize models
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

    loss = AdversarialLoss(generator, discriminator)

    train_loader = itertools.cycle(train_loader)
    test_loader = itertools.cycle(test_loader)

    # training
    for step in range(config.steps):
        # train discriminator
        for _ in range(config.k_steps):
            noise = torch.randn(config.batch_size, config.latent_dim).to(device)
            batch = next(iter(train_loader))[0].to(device)
            # normalize
            batch = (batch - mean_tr) / std_tr
            # backprop
            loss_disc = loss.discriminator_loss(batch, noise)
            loss_disc.backward()
            opt_disc.step()

        # train generator
        noise = torch.randn(config.batch_size, config.latent_dim).to(device)
        loss_gen = loss.generator_loss(noise)
        loss_gen.backward()
        opt_gen.step()

        if step % config.log_interval == 0:
            noise = torch.randn(config.batch_size, config.latent_dim).to(device)
            g_out = generator(noise)
            sample_generator(g_out, config.batch_size)
            # log metrics
            logging.info(
                f"Step {step} | G_loss: {loss_gen:>4f} | D_loss: {loss_disc:>4f}"
            )
            wandb.log(
                {
                    "gen_loss": loss_gen,
                    "disc_loss": loss_disc,
                }
            )


if __name__ == "__main__":
    app.run(main)
