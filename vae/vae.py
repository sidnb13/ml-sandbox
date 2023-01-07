"""Implementation of a variational autoencoder (VAE) in PyTorch on MNIST data."""

import logging

import config
import torch
from absl import flags
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
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.lin1(x))
        mean = self.lin2(h)
        logvar = self.lin3(h)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_classes=10) -> None:
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.lin1(x))
        out = F.relu(self.lin2(out))
        return self.lin3(out)


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=2) -> None:
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, num_classes=10)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(logvar) * torch.exp(logvar / 2) + mean

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        mean, logvar = self.encoder(data)
        # latent distribution is N(mean, var)
        z = self.reparameterize(mean, logvar)
        return z, self.decoder(z)


def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    # based on section B, Gaussian case
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - torch.exp(log_var))

def loss_function(recon_x, x, mu, log_var):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kl_loss = kl_divergence(mu, log_var)
    return kl_loss, recon_loss, kl_loss + recon_loss


def train_step():
    pass


def eval_step():
    pass


if __name__ == "__main__":
    logging.info("Loading data...")
    # train_loader = DataLoader(
    #     datasets.MNIST(
    #         config.data_dir,
    #         train=True,
    #         download=True,
    #         transform=transforms.ToTensor(),
    #     ),
    #     batch_size=config.batch_size,
    #     shuffle=True,
    # )
    # test_loader = DataLoader(
    #     datasets.MNIST(
    #         config.data_dir,
    #         train=False,
    #         download=True,
    #         transform=transforms.ToTensor(),
    #     ),
    #     batch_size=config.batch_size,
    #     shuffle=True,
    # )

    # logging.info("Building model...")
    # model = VAE().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)