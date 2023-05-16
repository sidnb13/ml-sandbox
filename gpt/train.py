"""
Small decoder-only transfomer for autoregressive text generation.
"""

import os
from typing import Any, Iterator

import psutil
import torch
import wandb
from absl import app, flags, logging
from datasets import Dataset as HFDataset
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast

FLAGS = flags.FLAGS

# define command line flags
flags.DEFINE_boolean(
    "use_wandb",
    False,
    "Whether to use wandb.",
)
flags.DEFINE_string("wandb_entity", None, "Wandb entity.", required=True)
flags.DEFINE_string("wandb_project", None, "Wandb project.", required=True)
flags.DEFINE_enum(
    "run_mode",
    "train",
    ["train", "test", "generate"],
    "Mode to run GPT model in.",
)
flags.DEFINE_string(
    "prompt",
    "test",
    "Prompt to use for autoregressive generation.",
)
flags.DEFINE_integer(
    "batch_size",
    100,
    "Batch size (each with seq_len tokens).",
    lower_bound=1,
)
flags.DEFINE_integer("steps", 10000, "Number of training steps.")
flags.DEFINE_integer("embed_dim", 128, "Embedding dimension, a power of 2.")
flags.DEFINE_integer("heads", 2, "Number of attention heads.")
flags.DEFINE_integer("blocks", 3, "Number of transformer blocks.")
flags.DEFINE_float("dropout", 0.1, "Dropout probability.")
flags.DEFINE_integer(
    "seq_len",
    128,
    "Sequence length for training in tokens.",
    lower_bound=4,
    upper_bound=1024,
)
flags.DEFINE_integer("gen_len", 128, "Sequence length for generation in tokens.")


def setup_device() -> torch.device:
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


def create_train_tokenizer(
    raw_dataset: HFDataset,
    tokenizer_file_path: str,
    batch_size: int = 1000,
) -> PreTrainedTokenizerFast:
    if not os.path.exists(tokenizer_file_path):
        # create the tokenizer
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        )
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.normalizer = BertNormalizer()

        # train tokenizer in batches
        def batch_iterator():
            for i in range(0, len(raw_dataset), batch_size):
                yield raw_dataset[i : i + batch_size]["text"]

        tokenizer.train_from_iterator(
            batch_iterator(), trainer=trainer, length=len(raw_dataset)
        )
        tokenizer.save(tokenizer_file_path)
    else:
        tokenizer = Tokenizer.from_file(tokenizer_file_path)
    # wrap in a fast tokenizer for reuse
    wrapped_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    return wrapped_tokenizer


class SimpleTextDataset(Dataset):
    def __init__(
        self,
        path: str,
        name: str,
        split: str,
        block_size: int,
        tokenizer_batch_size: int = 1000,
    ) -> None:
        raw_dataset = load_dataset(path, name, split=split)
        # remove empty rows
        raw_dataset = raw_dataset.filter(lambda x: len(x["text"]) > 0)

        self.tokenizer = create_train_tokenizer(
            raw_dataset,
            tokenizer_file_path=f"{path}-tokenizer.json",
            batch_size=tokenizer_batch_size,
        )

        if os.path.exists(f"{path}-{split}-tokenized.pt"):
            self.data = torch.load(f"{path}-{split}-tokenized.pt")
        else:
            encoded = raw_dataset.map(
                lambda x: self.tokenizer(x["text"]),
                batched=True,
            )
            # save as a series of tokens, discarding sentence structure
            tokens = torch.cat([torch.tensor(x) for x in encoded["input_ids"]])
            self.data = torch.tensor(tokens, dtype=torch.long)
            torch.save(self.data, f"{path}-{split}-tokenized.pt")

        self.block_size = block_size

    def encode(self, block_input: str) -> torch.Tensor:
        return self.tokenizer(block_input)["input_ids"]

    def decode(self, block_ids: torch.Tensor) -> str:
        return self.tokenizer.decode(block_ids)

    def tokenize(self, block_input: str) -> list[str]:
        return self.tokenizer.tokenize(block_input)

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, index) -> Any:
        # Consider block_size + 1 tokens from the dataset, starting from index.
        return {
            "input": self.data[index : index + self.block_size],
            "target": self.data[index + 1 : index + self.block_size + 1],
        }


class DecoderBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        attn_dropout: float,
        layer_dropout: float,
        return_attn_weights: bool = False,
    ) -> None:
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attn_dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        # positionwise FFN
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),  # as in original GPT
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(layer_dropout),
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.return_attn_weights = return_attn_weights

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # self attention and residual
        T = x.size(1)
        causal_mask = torch.tril(torch.ones(T, T)).to(x.device)
        attn_out, attn_weights = self.multihead_attention(
            x, x, x, attn_mask=causal_mask
        )
        x = self.ln1(x + attn_out)
        # feed forward and residual
        x = self.ln2(x + self.feed_forward(x))

        if self.return_attn_weights:
            return x, attn_weights
        else:
            return x


class Transformer(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        blocks: int,
        num_heads: int,
        embed_dim: int,
        attn_dropout: float,
        layer_dropout: float,
    ) -> None:
        """Transformer based on decoder blocks.

        Args:
            num_tokens (int): the number of tokens per input sequence.
            blocks (int): number of decoder blocks to stack.
            num_heads (int): number of attention heads.
            embed_dim (int): the embedding dimension.
            attn_dropout (float): the dropout rate for attention layers.
            layer_dropout (float): the dropout rate for the residual layers.
        """
        super().__init__()
        self.stack_count = blocks
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        # word embedding layers
        self.embedding = nn.Embedding(num_tokens, embed_dim)
        self.pos_embedding = nn.Embedding(num_tokens, embed_dim)
        self.positional_encoding(self.pos_embedding.weight)
        # decoder stack
        self.decoder_stack = nn.Sequential(
            *[
                DecoderBlock(
                    num_heads,
                    embed_dim,
                    attn_dropout=attn_dropout,
                    layer_dropout=layer_dropout,
                    return_attn_weights=(i == blocks - 1),
                )
                for i in range(blocks)
            ]
        )
        self.dropout = nn.Dropout(layer_dropout)
        # linear layer
        self.linear = nn.Linear(embed_dim, num_tokens)
        self.ln = nn.LayerNorm(embed_dim)
        # weight tie between embedding and linear weights
        self.linear.weight = self.embedding.weight

    def positional_encoding(self, embed_weights: torch.Tensor):
        # create positional encodings
        pos = torch.arange(0, self.num_tokens, dtype=torch.float)
        pos_encoding = torch.stack(
            [
                pos / pow(10000, (2 * dim // 2) / self.embed_dim)
                for dim in range(self.embed_dim)
            ],
        ).transpose_(0, 1)
        # assign positional encodings as non-trainable weights
        embed_weights.requires_grad = False
        embed_weights[:, 0::2] = torch.sin(pos_encoding[:, 0::2])
        embed_weights[:, 1::2] = torch.cos(pos_encoding[:, 1::2])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # create embeddings
        tok_embed = self.embedding(idx) * self.embed_dim**0.5
        pos_embed = self.pos_embedding(idx)
        input_embed = tok_embed + pos_embed
        # feed to transformer
        x = self.dropout(input_embed)
        x, attn_weights = self.decoder_stack(x)
        x = self.ln(x)
        x = self.linear(x)
        return x, attn_weights


def dumb_baseline(config: dict):
    """Make sure the model is initialized correctly."""
    device = setup_device()
    # get some basic model stats
    gpt = Transformer(**config).to(device)
    num_params = sum(p.numel() for p in gpt.parameters())
    gpt.eval()
    sample = torch.zeros(1, 4, dtype=torch.long, device=device)
    # check loss at init
    loss = Trainer.loss_fn(gpt, {"input": sample, "target": sample}, device)
    random_init_loss = -torch.log(torch.tensor(1 / gpt.num_tokens))
    logging.info(f"Number of parameters: {num_params}")
    logging.info(f"loss: {loss} | random_init: {random_init_loss}")
    assert loss.item() - random_init_loss < 1e-5, "softmax loss check failed"


class NoamOpt:
    def __init__(
        self,
        parameters: Iterator[torch.nn.Parameter],
        lr: float = 0,
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
        self.opt = optim.Adam(parameters, lr=lr, betas=betas, eps=eps)
        self.embed_dim = embed_dim
        self.warmup_steps = warmup_steps
        self._step = 0

    def zero_grad(self):
        self.opt.zero_grad()

    def step(self):
        self._step += 1
        self._update_lr()
        self.opt.step()

    def _lr(self):
        # lrate = d^−0.5 * min(step_num−0.5, step_num * warmup_steps^−1.5)
        return self.embed_dim**-0.5 * min(
            self._step**-0.5, self._step * self.warmup_steps**-1.5
        )

    def _update_lr(self):
        for param_group in self.opt.param_groups:
            param_group["lr"] = self._lr()


class Trainer:
    def __init__(self, model: Transformer, device: torch.device) -> None:
        """Training class to centralize training and validation logic.

        Args:
            model (Transformer): the model to train.
            device (torch.device): device to use for training.
        """
        self.model = model.to(device)
        self.opt = NoamOpt(
            model.parameters(),
            lr=0,
            betas=(0.9, 0.98),
            eps=1e-9,
            embed_dim=model.embed_dim,
            warmup_steps=4000,
        )
        self.device = device

    def loss_fn(
        self,
        batch: dict[str, torch.Tensor],
    ):
        input_ids = batch["input"].to(self.device)
        target_ids = batch["target"].to(self.device)
        logits, _ = self.model(input_ids)
        return F.cross_entropy(logits, target_ids)

    def train_step(self, batch: dict[str, torch.Tensor]):
        self.model.train()
        self.opt.zero_grad()
        # perform train step
        loss = self.loss_fn(batch)
        loss.backward()
        self.opt.step()
        return loss.item()

    def eval_step(self, batch: dict[str, torch.Tensor]):
        self.model.eval()
        with torch.no_grad():
            loss = self.loss_fn(batch, self.device)
        return loss.item()


def train(train_loader: DataLoader, trainer: Trainer, steps: int):
    for step in range(steps):
        # train
        batch = next(iter(train_loader))
        loss = trainer.train_step(batch)
        # eval
        if step % 100 == 0:
            logging.info(
                f"step: {step:07} | loss: {loss:3f} | lr: {trainer.opt._lr():.6f}"
            )
            if FLAGS.use_wandb:
                wandb.log({"loss": loss, "lr": trainer.opt._lr(), "step": step})


def main(argv):
    del argv  # Unused.

    if FLAGS.use_wandb:
        wandb.init(
            settings=wandb.Settings(start_method="fork"),
            project=FLAGS.wandb_project,
            entity=FLAGS.wandb_entity,
            config=FLAGS.flag_values_dict(),
        )

    # TODO: add other modes (generation, finetune, etc.)
    if FLAGS.run_mode == "train":
        # create datasets
        dataset_train = SimpleTextDataset(
            "wikitext", "wikitext-2-raw-v1", "train", FLAGS.seq_len
        )
        train_loader = DataLoader(
            dataset_train, batch_size=FLAGS.batch_size, shuffle=True
        )
        # create model
        # TODO: centralize and streamline model config
        model = Transformer(
            FLAGS.seq_len,
            FLAGS.blocks,
            FLAGS.heads,
            FLAGS.embed_dim,
            FLAGS.dropout,
            FLAGS.dropout,
        )
        # create trainer
        trainer = Trainer(model, setup_device())
        # train
        train(train_loader, trainer, FLAGS.steps)


if __name__ == "__main__":
    app.run(main)
