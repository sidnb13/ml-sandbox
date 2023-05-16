"""
Implementation of an small decoder-only transfomer which autoregressively generates text.
"""

import os
import sys
from typing import Any

import psutil
import torch
from absl import app, flags, logging
from datasets import Dataset as HFDataset
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch import nn
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

FLAGS = flags.FLAGS

# define command line flags
flags.DEFINE_boolean(
    "use_wandb",
    False,
    "Whether to use wandb.",
)
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
        self, num_heads: int, embed_dim: int, attn_dropout: float, layer_dropout: float
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

    def forward(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        del kwargs
        # self attention and residual
        T = x.size(1)
        causal_mask = torch.tril(torch.ones(T, T)).to(x.device)
        attn_out, attn_weights = self.multihead_attention(
            x, x, x, attn_mask=causal_mask
        )
        x = self.ln1(x + attn_out)
        # feed forward and residual
        x = self.ln2(x + self.feed_forward(x))
        return x, attn_weights


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
                )
                for _ in range(blocks)
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
        x = self.embedding(idx) + self.pos_embedding(idx)
        x = self.dropout(x)
        x, attn_weights = self.decoder_stack(x)
        x = self.ln(x)
        x = self.linear(x)
        return x, attn_weights


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


def main(argv):
    del argv  # Unused.

    # dataset = SimpleTextDataset("wikitext", "wikitext-2-raw-v1", "train", 4)
    # print(dataset.decode(dataset[0]["input"]))

    device = setup_device()

    gpt = Transformer(4, 1, 1, 128, 0.1, 0.1).to(device)

    print("num params", sum(p.numel() for p in gpt.parameters()))

    out, _ = gpt(torch.zeros(1, 4, dtype=torch.long, device=device))

    print(out)

    logging.info(
        "Running under Python {0[0]}.{0[1]}.{0[2]}".format(sys.version_info),
    )


if __name__ == "__main__":
    app.run(main)
