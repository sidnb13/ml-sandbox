"""ResNet implementation in PyTorch."""
from datetime import datetime
from typing import *

import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from absl import app, flags, logging
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import datasets
from tqdm import tqdm

FLAGS = flags.FLAGS
job_id = datetime.strftime(datetime.now(), "%m-%d-%y-%H-%M-%S")

flags.DEFINE_string(
    "plot_name",
    "resnet-{}.png".format(job_id),
    "Name of the plot to save.",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        pre_expansion_channels,
        stride=1,
        use_batchnorm=True,
        expansion=1,
    ) -> None:
        super().__init__()

        self.expansion = expansion
        self.NUM_CONV_LAYERS = 2

        self.conv1 = nn.Conv2d(
            in_channels, pre_expansion_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = (
            nn.BatchNorm2d(pre_expansion_channels) if use_batchnorm else nn.Identity()
        )
        self.conv2 = nn.Conv2d(
            pre_expansion_channels,
            pre_expansion_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = (
            nn.BatchNorm2d(pre_expansion_channels) if use_batchnorm else nn.Identity()
        )
        self.relu = nn.ReLU()

        if in_channels != pre_expansion_channels * expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    pre_expansion_channels * expansion,
                    kernel_size=1,
                    stride=stride,
                ),
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
        self.NUM_CONV_LAYERS = 3

        # No batchnorm applied to the first conv layer
        self.conv1 = (
            nn.Conv2d(in_channels, pre_expansion_channels, kernel_size=1, stride=1)
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
            pre_expansion_channels,
            pre_expansion_channels * expansion,
            kernel_size=1,
            stride=1,
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
    def __init__(self, in_channels=16, use_batchnorm=True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.NUM_CONV_LAYERS = 1

        # Pre-residual conv layer
        self.conv1 = nn.Conv2d(3, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU()

        # Residual blocks
        self.layer1 = self._create_layer(ResidualBlock, 16, num_blks=1, stride=1)
        self.layer2 = self._create_layer(ResidualBlock, 32, num_blks=1, stride=2)
        self.layer3 = self._create_layer(ResidualBlock, 64, num_blks=1, stride=2)

        # Linear output
        self.readout = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, config.num_classes),
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

        for stride_val in stride_pattern:
            blk = block(self.in_channels, channels, stride=stride_val)
            layers.append(blk)
            self.NUM_CONV_LAYERS += blk.NUM_CONV_LAYERS
            self.in_channels = channels * blk.expansion

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        return self.readout(out)


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


def train_loop(model, dataloader_train, dataloader_test, criterion, optimizer):
    model.train()

    logging.info(
        "Training for {:4d} epochs with lr={:.6f}".format(config.epochs, config.lr)
    )
    
    train_losses, train_accs, eval_losses, eval_accs = [], [], [], []

    for epoch in range(config.epochs):
        total_loss = 0
        total_correct = 0

        for step, (data, target) in tqdm(
            enumerate(dataloader_train), unit="batch", total=len(dataloader_train), leave=False
        ):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            total_correct += torch.sum(torch.argmax(output, dim=1) == target)

            if step % config.checkpt_interval == 0:
                                
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "loss": loss,
                        "step": step,
                        "epoch": epoch,
                    },
                    config.checkpt_dir / "checkpt_{}.pt".format(job_id),
                )
                logging.debug(
                    "Saved checkpoint at step {}, epoch {}".format(step, epoch)
                )

        if epoch % config.log_interval == 0:
            train_loss = total_loss / len(dataloader_train)
            train_acc = total_loss / len(dataloader_train)
            logging.info(
                "Epoch: [{}/{}]\tLoss: {:.6f}\tAcc: {:.4f}".format(
                    epoch,
                    config.epochs,
                    train_loss,
                    train_acc,
                    correct_preds / len(data),
                )
            )
            
            train_losses.append(train_loss)
            train_accs.append(train_acc)

        if step % config.eval_interval == 0:
            eval_loss, eval_acc = compute_metrics(dataloader_test, model, criterion)
            logging.info(
                "Eval: [{}/{}]\tLoss: {:.6f}\tAcc: {:.4f}".format(
                    epoch,
                    config.epochs,
                    eval_loss,
                    eval_acc,
                )
            )
            
            eval_losses.append(eval_loss)
            eval_accs.append(eval_acc)
    
    save_plot(model, train_losses, train_accs, eval_losses, eval_accs)


def compute_metrics(dataloader, model, criterion):
    model.eval()

    total_loss = 0
    total_correct = 0

    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()
        total_correct += torch.sum(torch.argmax(output, dim=1) == target)

    return total_loss / len(dataloader), total_correct / len(dataloader.dataset)


def save_plot(model: ResNet, train_loss, train_acc, eval_loss, eval_acc):
    plt.plot(train_loss, label="Train")
    plt.plot(eval_loss, label="Eval")
    plt.plot(train_acc, label="Train")
    plt.plot(eval_acc, label="Eval")
    plt.set_title("ResNet-{} {}".format(model.NUM_CONV_LAYERS, job_id))
    plt.legend()
    plt.savefig(os.path.join(config.save_dir, FLAGS.plot_name))


def main(argv):
    del argv  # unused

    # set up CIFAR10 dataset
    mean_std = [(0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)]
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(*mean_std),
            transforms.RandomCrop(32, 4),
        ]
    )
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(*mean_std)]
    )

    train_set = datasets.CIFAR10(
        config.data_dir, train=True, download=True, transform=train_transform
    )
    test_set = datasets.CIFAR10(
        config.data_dir, train=False, download=True, transform=test_transform
    )

    # set up data loader
    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True, drop_last=False
    )
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=False, drop_last=False
    )

    # configure training
    model = ResNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # model summary
    summary(model)
    print(model)
        
    train_loop(model, dataloader_train, dataloader_test, criterion, optimizer)
    

if __name__ == "__main__":
    app.run(main)
