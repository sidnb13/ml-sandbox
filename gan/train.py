import itertools
import os
from datetime import datetime

import config
import numpy as np
import torch
import wandb
from absl import app, flags, logging
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu")
FLAGS = flags.FLAGS
flags.DEFINE_boolean("use_wandb", False, "Use wandb for experiment logging.")

timestamp = torch.tensor(datetime.now().timestamp()).to(device)
timestamp = datetime.fromtimestamp(timestamp.int()).strftime("%Y-%m-%d-%H-%M-%S")


class Generator(nn.Module):
    def __init__(self, latent_dim: int, image_dim: int, act: str = "ReLU") -> None:
        super().__init__()
        self.act = act
        self.latent_dim = latent_dim
        self.image_dim = image_dim

        lin_list = torch.nn.ModuleList()
        prev = self.latent_dim

        for hidden_dim in config.hidden_dims:
            lin_list.append(create_block(prev, hidden_dim, self.act))
            prev = hidden_dim

        lin_list.append(create_block(hidden_dim, self.image_dim, "Tanh"))

        self.lin_list = nn.Sequential(*lin_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin_list(x)


class Discriminator(nn.Module):
    def __init__(self, in_dim: int, act: str = "ReLU") -> None:
        super().__init__()
        self.act = act
        self.in_dim = in_dim

        lin_list = []
        prev = self.in_dim

        for hidden_dim in config.hidden_dims[::-1]:
            lin_list.append(create_block(prev, hidden_dim, self.act))
            prev = hidden_dim

        lin_list.append(create_block(hidden_dim, 1, "Sigmoid"))

        self.lin_list = nn.Sequential(*lin_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin_list(x)


def create_block(in_dim: int, out_dim: int, act) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        getattr(nn, act)() if act is not None else nn.Identity(),
    )


def sample_generator(g_out: torch.Tensor, n: int) -> torch.Tensor:
    path = f"{config.save_dir}/gen-{timestamp}.png"
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    save_image(
        g_out[:n].view(-1, 1, *config.image_size),
        path,
        nrow=n,
    )


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
        drop_last=True,
    )
    test_loader = DataLoader(
        datasets.MNIST(
            config.data_dir,
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        ),
        batch_size=config.batch_size,
        drop_last=True,
    )

    # normalization
    mean_tr, mean_ts = 0.0, 0.0
    std_tr, std_ts = 0.0, 0.0

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

    train_loader = itertools.cycle(train_loader)
    test_loader = itertools.cycle(test_loader)

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

    criterion = torch.nn.BCELoss()

    # fixed noise for sampling
    fixed_noise = torch.randn(config.batch_size, config.latent_dim).to(device)

    # training
    for step in range(config.steps):
        # train discriminator
        discriminator.train()

        p_real, p_fake = 0, 0

        for _ in range(config.k_steps):
            opt_disc.zero_grad()
            noise = torch.randn(config.batch_size, config.latent_dim).to(device)

            batch = next(iter(train_loader))[0]

            # prepare batch and compute outputs
            batch = (batch - mean_tr) / std_tr
            batch = batch.view(-1, np.prod(config.image_size))
            # D(x)
            disc_out_real = discriminator(batch.to(device))
            # D(G(z))
            disc_out_fake = discriminator(generator(noise))

            # train on real data, compare to ground truth, log(D(x))
            loss_disc_real = criterion(disc_out_real, torch.ones_like(disc_out_real))

            # train on fake data, compare to ground truth, log(1 - D(G(z)))
            loss_disc_fake = criterion(disc_out_fake, torch.zeros_like(disc_out_fake))

            loss_disc = (loss_disc_fake + loss_disc_real) / 2
            loss_disc.backward()

            # discriminator gradient update
            opt_disc.step()

            # update probabilities
            p_real += torch.mean(disc_out_real)
            p_fake += torch.mean(disc_out_fake)

        p_real /= config.k_steps
        p_fake /= config.k_steps

        # train generator
        generator.train()
        opt_gen.zero_grad()
        noise = torch.randn(config.batch_size, config.latent_dim).to(device)

        gen_out = generator(noise)

        # log(1 - D(G(z)))
        disc_gen = discriminator(gen_out)
        loss_gen = criterion(disc_gen, torch.zeros_like(disc_gen))
        loss_gen.backward()
        opt_gen.step()

        if step % config.log_interval == 0:
            with torch.no_grad():
                g_out = generator(fixed_noise)
                sample_generator(g_out, config.batch_size)
                # log metrics
                loss_disc = loss_disc_fake + loss_disc_real
                logging.info(
                    f"[Step {step}]\tG_loss: {loss_gen:>4f}\tD_loss: {loss_disc:>4f}\tD(x): {p_real:>4f}\t D(G(z)): {p_fake:>4f}"
                )
                if FLAGS.use_wandb:
                    wandb.log(
                        {
                            "G_loss": loss_gen,
                            "D_loss": loss_disc,
                            "D(G(z))": p_fake,
                            "D(x)": p_real,
                        }
                    )


if __name__ == "__main__":
    app.run(main)
