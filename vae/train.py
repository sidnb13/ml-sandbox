"""Implementation of a variational autoencoder (VAE) in PyTorch on MNIST data."""

import itertools
import logging
import os
import pathlib

import config
import matplotlib.pyplot as plt
import numpy as np
import torch
from absl import app, flags
from colorama import Fore, Style
from scipy.stats import norm
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

logging.basicConfig(level=logging.INFO)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "plot_name",
    f"vae-{config.job_id}.png",
    "Name of the plot to save.",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinearEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim) -> None:
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, latent_dim)
        self.lin3 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.lin1(x))
        mean = self.lin2(h)
        logvar = self.lin3(h)
        return mean, logvar


class LinearDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim) -> None:
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.lin1(x))
        return self.lin2(out)


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim) -> None:
        super().__init__()
        self.encoder = LinearEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = LinearDecoder(input_dim, hidden_dim, latent_dim)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(logvar)
        z = mean + eps * torch.exp(logvar * 0.5)
        return z, mean, logvar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, mean, logvar = self.reparameterize(*self.encoder(x))
        recon_x = self.decoder(z)
        return z, recon_x, mean, logvar


def loss_fn(x, recon_x, mu, log_var):
    kl_divergence = -0.5 * torch.sum(
        1 + log_var - torch.square(mu) - torch.exp(log_var)
    )
    recon_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction="sum")
    return kl_divergence + recon_loss, kl_divergence, recon_loss


def train_step(batch, model, optimizer):
    model.train()
    x = batch[0].to(device).reshape(-1, config.input_dim)
    optimizer.zero_grad()
    # take grads with respect to total loss
    z, recon_x, mean, logvar = model(x)
    total, kl, recon = loss_fn(x, recon_x, mean, logvar)

    total.backward()
    optimizer.step()

    return {
        "kl": kl.item() / len(x),
        "recon": recon.item() / len(x),
        "total": total.item() / len(x),
        "recon_x": recon_x,
        "mu": mean,
        "logvar": logvar,
        "z": z,
    }


def eval_step(batch, model):
    model.eval()
    x = batch[0].to(device).reshape(-1, config.input_dim)
    z, recon_x, mean, logvar = model(x)
    total, kl, recon = loss_fn(x, recon_x, mean, logvar)

    return {
        "kl": kl.item() / len(x),
        "recon": recon.item() / len(x),
        "total": total.item() / len(x),
        "recon_x": recon_x,
        "mu": mean,
        "logvar": logvar,
        "z": z,
    }


def main(argv):
    del argv
    logging.info("Loading data...")

    train_loader = DataLoader(
        datasets.MNIST(
            config.data_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        ),
        batch_size=config.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        datasets.MNIST(
            config.data_dir,
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        ),
        batch_size=config.batch_size,
        shuffle=False,
    )

    logging.info("Building model...")
    model = VAE(config.input_dim, config.hidden_dim, config.latent_dim)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # train loop
    train_iter = itertools.cycle(train_loader)

    logging.info(
        f"Begin training for {config.steps:6d} steps with lr={config.lr:10.5f}..."
    )

    # clean the results folder
    if pathlib.Path(config.result_dir).exists():
        imgs = pathlib.Path(config.result_dir)
        for path in imgs.glob("*.png"):
            path.unlink()

    for step in range(config.steps):
        batch = next(train_iter)
        out = train_step(batch, model, optimizer)

        _kl = out["kl"]
        _recon = out["recon"]
        _total = out["total"]

        if step % config.log_interval == 0:
            logging.info(
                f"[Step {step:06d}]\tloss: {_total:10.5f}\t"
                + f"kl: {_kl:10.5f}\trecon: {_recon:10.5f}"
            )

        if step % config.test_interval == 0:
            kl, recon, total = 0, 0, 0

            for test_batch in test_loader:
                eval_out = eval_step(test_batch, model)
                kl += eval_out["kl"]
                recon += eval_out["recon"]
                total += eval_out["total"]

            logging.info(
                f"{Fore.GREEN}[Step {step:06d}]{Style.RESET_ALL}\t"
                + f"loss: {total / len(test_loader):10.5f}\t"
                + f"kl: {kl / len(test_loader):10.5f}\trecon: {recon / len(test_loader):10.5f}"
            )

            # sample reconstructions
            comparison = torch.cat(
                [
                    batch[0][: config.gen_samples],
                    out["recon_x"].view(config.batch_size, 1, 28, 28)[
                        : config.gen_samples
                    ],
                ]
            )
            path = f"{config.result_dir}/generated_{step // config.test_interval}.png"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            save_image(
                comparison.cpu(),
                path,
                nrow=config.gen_samples,
            )

            rand_sample_idx = np.random.randint(0, config.batch_size)

            # plot latent Gaussian
            fig, axs = plt.subplots(config.latent_dim, sharex=True, sharey=True)
            fig.suptitle(
                f"Latent Gaussian Distributions {step // config.test_interval}"
            )

            mu, sigma = (
                out["mu"][rand_sample_idx].cpu().detach().numpy(),
                torch.exp(0.5 * out["logvar"][rand_sample_idx]).cpu().detach().numpy(),
            )

            for i, ax in enumerate(axs):
                x = np.linspace(mu[i] - 3 * sigma[i], mu[i] + 3 * sigma[i], 100)
                pdf = norm(mu[i], sigma[i]).pdf(x)
                ax.plot(x, pdf)

            fig.savefig(f"{config.result_dir}/zdist_{step // config.test_interval}.png")

            # create manifold
            if config.latent_dim == 2:
                dim = int(np.sqrt(len(batch[0])))
                img_grid = np.zeros((28 * dim, 28 * dim))

                inverse_norm = norm.ppf(np.linspace(0.05, 0.95, dim))

                with torch.no_grad():
                    for i, x_inc in enumerate(inverse_norm):
                        for j, y_inc in enumerate(inverse_norm):
                            z = torch.Tensor([x_inc, y_inc], device=device)
                            img_grid[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = (
                                model.decoder(z).cpu().detach().numpy().reshape(28, 28)
                            )

            save_image(
                torch.tensor(img_grid),
                f"{config.result_dir}/manifold_{step // config.test_interval}.png",
            )


if __name__ == "__main__":
    app.run(main)
