"""
Implementation of an small causal/autoregressive transformer (decoder-only architecture), GPT-style.
"""

import os
import sys

from absl import app, flags, logging
from typing import Any

from datasets import Dataset as HFDataset, DatasetDict, load_dataset, load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

import torch
from torch.utils.data import DataLoader, Dataset

FLAGS = flags.FLAGS

flags.DEFINE_string("echo", None, "Text to echo.")


def create_train_tokenizer(
    raw_dataset: HFDataset,
    tokenizer_file_path: str,
    batch_size: int = 1000,
) -> PreTrainedTokenizerFast:
    """Create a tokenizer from a raw dataset.

    Args:
        raw_dataset (HFDataset): _description_
        tokenizer_file_path (str): _description_
        batch_size (int, optional): _description_. Defaults to 1000.

    Returns:
        PreTrainedTokenizerFast: _description_

    Yields:
        Iterator[PreTrainedTokenizerFast]: _description_
    """
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
        """A simple text dataset for causal/autoregressive models.

        Args:
            path (str): dataset path (e.g. "wikitext-2")
            name (str): the name of the dataset
            split (str): split of the dataset (e.g. "train")
            block_size (int): The maximum length of a sequence in tokens.
            tokenizer_batch_size (int, optional): Defaults to 1000.
        """
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


def main(argv):
    del argv  # Unused.

    dataset = SimpleTextDataset("wikitext", "wikitext-2-raw-v1", "train", 4)

    print(dataset.decode(dataset[0]["input"]))

    print(
        "Running under Python {0[0]}.{0[1]}.{0[2]}".format(sys.version_info),
        file=sys.stderr,
    )
    logging.info("echo is %s.", FLAGS.echo)


if __name__ == "__main__":
    app.run(main)
