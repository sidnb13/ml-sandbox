import os
from typing import Any, Iterator

import psutil
import requests
import tiktoken
import torch
from absl import logging
from ml_collections import config_dict
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset


class NoamOpt:
    def __init__(
        self,
        parameters: Iterator[torch.nn.Parameter],
        betas: tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-9,
        embed_dim: int = 512,
        warmup_steps: int = 4000,
    ) -> None:
        """Implements the Noam learning rate schedule.

        Args:
            parameters (Iterator[torch.nn.Parameter]): the model parameters.
            lr (float, optional): initial lr. Defaults to 0.
            betas (tuple[float, float], optional): betas for Adam. Defaults to (0.9, 0.98).
            eps (float, optional): eps for Adam. Defaults to 1e-9.
            embed_dim (int, optional): model embedding dimension. Defaults to 512.
            warmup_steps (int, optional): warmup steps prior to decay. Defaults to 4000.
        """
        self._opt = optim.Adam(parameters, betas=betas, eps=eps)
        self.embed_dim = embed_dim
        self.warmup_steps = warmup_steps
        self._step = 0

    def zero_grad(self):
        self._opt.zero_grad()

    def step(self):
        self._step += 1
        self._update_lr()
        self._opt.step()

    def _lr(self):
        # lrate = d^−0.5 * min(step_num−0.5, step_num * warmup_steps^−1.5)
        return self.embed_dim**-0.5 * min(
            self._step**-0.5, self._step * self.warmup_steps**-1.5
        )

    def _update_lr(self):
        for param_group in self._opt.param_groups:
            param_group["lr"] = self._lr()


class Trainer:
    def __init__(self, model, config, device: torch.device) -> None:
        """Training class to centralize training and validation logic.

        Args:
            model: the model to train.
            device (torch.device): device to use for training.
        """
        self.model = model.to(device)
        self.opt = NoamOpt(
            model.parameters(),
            betas=config.betas,
            eps=config.eps,
            embed_dim=model.embed_dim,
            warmup_steps=config.warmup_steps,
        )
        self.device = device

    @staticmethod
    def loss_fn(
        model,
        batch: dict[str, torch.Tensor],
        device: torch.device,
        bypass_embedding: bool = False,
    ):
        input_ids = batch["input"].to(device)
        target_ids = batch["target"].to(device)
        logits, _ = model(input_ids, bypass_embedding=bypass_embedding)
        return F.cross_entropy(logits.transpose(1, 2), target_ids)

    def train_step(self, batch: dict[str, torch.Tensor]):
        self.model.train()
        self.opt.zero_grad()
        # perform train step
        loss = Trainer.loss_fn(self.model, batch, self.device)
        loss.backward()
        self.opt.step()
        return loss.item()

    def eval_step(self, batch: dict[str, torch.Tensor]):
        self.model.eval()
        with torch.no_grad():
            loss = Trainer.loss_fn(self.model, batch, self.device)
        return loss.item()

    def save_checkpoint(self, path: str):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(
            {
                "model": self.model.state_dict(),
                "opt": self.opt._opt.state_dict(),
                "step": self.opt._step,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        state = torch.load(path)
        self.model.load_state_dict(state["model"])
        self.opt._opt.load_state_dict(state["opt"])
        self.opt._step = state["step"]


def setup_device() -> torch.device:
    """Determine which device to train on.
    Only supports single GPU training for now.

    Returns:
        torch.device: device to use for training.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logging.info(
            f"Using CUDA device: {torch.cuda.get_device_name()} ({torch.cuda.device_count()}x)"
        )
        logging.info(
            f"Memory available: {1e-9 * torch.cuda.mem_get_info(device)[0]:.1f} GB"
        )
        return device
    elif torch.backends.mps.is_available():
        logging.info("Using MPS device")
        return torch.device("mps")

    stats = psutil.virtual_memory()  # returns a named tuple
    available = getattr(stats, "available")
    logging.info(f"available memory: {1e-6 * available} mb")
    return torch.device("cpu")


class SimpleTextDataset(Dataset):
    def __init__(
        self,
        split: str,
        config: config_dict,
    ) -> None:
        """Construct a simple text dataset

        Args:
            path (str): path to dataset files
            split (str): which split to load
            config (config_dict): general config
        """

        self.config = config

        encoding = tiktoken.get_encoding("cl100k_base")

        self.encoding = encoding
        self.vocab_size = self.encoding.n_vocab
        self.block_size = config.hyperparams.block_size

        filename = os.path.join(os.path.dirname(__file__), "data.txt")
        if not os.path.exists(filename):
            with open(filename, "w") as f:
                f.write(requests.get(config.data_url).text)
        with open(filename, "r") as f:
            text = f.read()

        if split == "train":
            text = text[: int(config.train_split * len(text))]
        elif split == "val":
            text = text[int((1 - config.train_split) * len(text)) :]
        else:
            raise ValueError(f"Invalid split: {split}")

        save_path = os.path.join(
            os.path.dirname(__file__),
            "saved",
            f"{split}-{self.block_size}-tokenized.pt",
        )

        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        self.data = torch.tensor(encoding.encode_ordinary(text))

        if os.path.exists(save_path):
            self.data = torch.load(save_path)
        else:
            torch.save(self.data, save_path)

    def decode(self, block_ids: torch.Tensor) -> str:
        return self.encoding.decode(block_ids)

    def tokenize(self, block_input: str) -> list[str]:
        return self.encoding.encode(block_input)

    @property
    def num_tokens(self) -> int:
        return len(self.data)

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, index) -> Any:
        inputs = self.data[index : index + self.block_size]
        targets = self.data[index + 1 : index + self.block_size + 1]

        return {
            "input": inputs,
            "target": targets,
        }
