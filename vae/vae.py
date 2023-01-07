"""Implementation of a variational autoencoder (VAE) in PyTorch on MNIST data."""

import itertools
import logging

import config
import matplotlib.pyplot as plt
import torch
from absl import app, flags
from colorama import Fore, Style
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

logging.basicConfig(level=logging.INFO)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "plot_name",
    f"vae-{config.job_id}.png",
    "Name of the plot to save.",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
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


class Decoder(nn.Module):
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
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(logvar)
        z = mean + eps * torch.exp(logvar / 2)
        return z, mean, logvar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, mean, logvar = self.reparameterize(*self.encoder(x))
        recon_x = self.decoder(z)
        return z, recon_x, mean, logvar


def loss_fn(x, recon_x, mu, log_var):
    kl_divergence = -0.5 * torch.sum(1 + log_var - torch.square(mu) - torch.exp(log_var))
    recon_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction="sum")
    return kl_divergence, recon_loss


def train_step(batch, model, optimizer):
    model.train()
    x = batch[0].to(device).reshape(-1, config.input_dim) / 255.0
    optimizer.zero_grad()

    # take grads with respect to total loss
    _, recon_x, mean, logvar = model(x)
    kl, recon = loss_fn(x, recon_x, mean, logvar)

    kl = kl.mean()
    recon = recon.mean()
    total = kl + recon

    total.backward()
    optimizer.step()

    return kl.item(), recon.item(), total.item()


def eval_step(batch, model):
    model.eval()
    x = batch[0].to(device).reshape(-1, config.input_dim) / 255.0
    _, recon_x, mean, logvar = model(x)
    kl, recon = loss_fn(x, recon_x, mean, logvar)

    kl = kl.mean()
    recon = recon.mean()
    total = kl + recon

    return kl.item(), recon.item(), total.item()


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
        f"Begin training for {config.steps:6d} steps with lr={config.lr:.6f}..."
    )

    for step in range(config.steps):
        batch = next(train_iter)
        kl, recon, total = train_step(batch, model, optimizer)

        if step % config.log_interval == 0:
            logging.info(
                f"[Step {step:06d}]\tloss: {total:.5f}\tkl: {kl:.5f}\trecon: {recon:.5f}"
            )

        if step % config.test_interval == 0:
            kl, recon, total = 0, 0, 0

            for test_batch in test_loader:
                _kl, _recon, _total = eval_step(test_batch, model)
                kl += _kl
                recon += _recon
                total += _total
                
            logging.info(
                f"{Fore.GREEN}[Step {step:06d}]{Style.RESET_ALL}\tloss: {total / len(test_loader):.5f}\t"
                + f"kl: {kl / len(test_loader):.5f}\trecon: {recon / len(test_loader):.5f}"
            )

    # comparing original and generated images
    x = next(iter(test_loader))[0][5]
    x_recon = (
        model(x.to(device).reshape(-1, config.input_dim) / 255.0)[1]
        .detach()
        .cpu()
        .numpy()
        .reshape(28, 28)
    )

    # plot original and generated
    plt.subplot(1, 2, 1)
    plt.imshow(x.numpy().reshape(28, 28), cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(x_recon, cmap="gray")
    plt.savefig("vae_gen.png")


if __name__ == "__main__":
    app.run(main)
