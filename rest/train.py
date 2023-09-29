"""An implementation of Reinforced Self-Training (ReST)."""

from functools import partial
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app, flags
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ml_collections import config_dict, config_flags
from config import RestConfig, TrainingConfig, DatasetConfig, TransformerConfig

# define configuration file
config_flags.DEFINE_config_file(
    "config",
    None,
    "Config file for training.",
    lock_config=True,
)

flags.DEFINE_bool(
    "wandb",
    False,
    "Whether or not to log run to wandb",
)


class EncoderBlock(nn.Module):
    embedding_dim: int = 512
    ffn_dim: int = 1024
    num_heads: int = 4
    residual_dropout: float = 0.1
    attention_dropout: float = 0.1

    def setup(self) -> None:
        self.attn = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.embedding_dim,
            dropout_rate=self.attention_dropout,
        )
        self.mlp = nn.Sequential(
            [nn.Dense(self.ffn_dim), nn.gelu, nn.Dense(self.embedding_dim)]
        )
        self.dropout = nn.Dropout(self.residual_dropout)
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()

    def __call__(
        self,
        hidden_state: jnp.ndarray,
        attn_mask: jnp.ndarray,
        deterministic: bool = False,
    ) -> None:
        out = self.attn(hidden_state, mask=attn_mask, deterministic=deterministic)
        out = out + self.ln1(out)
        out = out + self.mlp(out)
        out = self.dropout(out, deterministic=deterministic)
        out = out + self.ln2(out)
        return out


class DecoderBlock(nn.Module):
    embedding_dim: int = 512
    ffn_dim: int = 1024
    num_heads: int = 4
    residual_dropout: float = 0.1
    attention_dropout: float = 0.1

    def setup(self) -> None:
        self.self_attn = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.embedding_dim,
            dropout_rate=self.attention_dropout,
        )
        self.mha = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embedding_dim,
            dropout_rate=self.attention_dropout,
        )
        self.mlp = nn.Sequential(
            [nn.Dense(self.ffn_dim), nn.gelu, nn.Dense(self.embedding_dim)],
        )
        self.dropout = nn.Dropout(self.residual_dropout)
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()

    def __call__(
        self,
        hidden_state: jnp.ndarray,
        context: jnp.ndarray,
        attn_mask: jnp.ndarray,
        causal_mask: jnp.ndarray,
        deterministic: bool = False,
    ) -> None:
        mask = jnp.dot(attn_mask, causal_mask)
        out = self.self_attn(hidden_state, mask=mask, deterministic=deterministic)
        out = out + self.ln1(out)
        out = self.mha(inputs_q=out, inputs_kv=context, deterministic=deterministic)
        out = out + self.mlp(out)
        out = self.dropout(out, deterministic=deterministic)
        out = self.ln2(out)
        return out


class Transformer(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.pos_embedding = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=1 / jnp.sqrt(self.config.embedding_dim)),
            (1, self.config.max_seq_len, self.config.embedding_dim),
        )
        self.embedding = nn.Embed(
            self.config.vocab_size,
            self.config.embedding_dim,
            embedding_init=nn.initializers.normal(
                stddev=1 / jnp.sqrt(self.config.embedding_dim)
            ),
        )
        self.encoder_blocks = [
            EncoderBlock(
                self.config.embedding_dim,
                self.config.ffn_dim,
                self.config.num_heads,
                self.config.residual_dropout,
                self.config.attention_dropout,
            )
            for _ in range(self.config.layers)
        ]

        self.decoder_blocks = [
            DecoderBlock(
                self.config.embedding_dim,
                self.config.ffn_dim,
                self.config.num_heads,
                self.config.residual_dropout,
                self.config.attention_dropout,
            )
            for _ in range(self.config.layers)
        ]

    def __call__(
        self,
        input_ids: jnp.ndarray,
        targets: jnp.ndarray,
        attn_mask: jnp.ndarray,
        deterministic: bool = False,
    ) -> jnp.ndarray:
        B, input_len = input_ids.shape
        _, target_len = targets.shape
        causal_mask = nn.make_causal_mask(jnp.ones((B, input_len)))
        emb_tokens = self.embedding(input_ids)
        emb_pos = self.pos_embedding[:, :input_len, :]
        ctx = emb_tokens + emb_pos

        for encoder_block in self.encoder_blocks:
            ctx = encoder_block(ctx, attn_mask, deterministic=deterministic)

        out = self.embedding(targets) + self.pos_embedding[:, :target_len, :]

        for decoder_block in self.decoder_blocks:
            out = decoder_block(
                ctx, out, attn_mask, causal_mask, deterministic=deterministic
            )

        logits = jnp.matmul(self.embedding.embedding, out)  # (B, V, E)
        return logits


class FlanDataset(Dataset):
    def __init__(self, split: str, dataset_config: DatasetConfig) -> None:
        super().__init__()
        self.input_column = dataset_config.input_column
        self.target_column = dataset_config.target_column
        self.max_seq_len = dataset_config.max_seq_len
        self._dataset = load_dataset(dataset_config.ds_name, split=split)
        self.reward_tokenizer = AutoTokenizer.from_pretrained(
            dataset_config.reward_tokenizer
        )
        self.reward_tokenizer.pad_token_id = self.reward_tokenizer.eos_token_id
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __getitem__(self, index: int) -> Any:
        item = self._dataset[index]
        input_ = item[self.input_column]
        target_ = item[self.target_column]
        input_kwargs = dict(
            text=input_,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        target_kwargs = dict(
            text=target_,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        input_model = self.tokenizer(**input_kwargs)
        target_model = self.tokenizer(**target_kwargs)
        input_reward = self.reward_tokenizer(**input_kwargs)
        target_reward = self.reward_tokenizer(**target_kwargs)

        return {
            "model": {
                "input_ids": input_model["input_ids"].squeeze(),
                "targets": target_model["input_ids"].squeeze(),
                "attention_mask": input_model["attention_mask"].squeeze(),
            },
            "reward": {
                "input_ids": input_reward["input_ids"].squeeze(),
                "targets": target_reward["input_ids"].squeeze(),
                "attention_mask": input_reward["attention_mask"].squeeze(),
            },
        }

    def __len__(self) -> int:
        return len(self._dataset)


def main(argv):
    del argv  # Unused.
    rng = jax.random.PRNGKey(0)
    key, param_rng, dropout_rng = jax.random.split(rng, 3)

    # ds = FlanDataset("train", DatasetConfig())
    # print(ds[0])

    model = Transformer(TransformerConfig())
    params = model.init(
        {"dropout": dropout_rng, "params": param_rng},
        jax.random.randint(key, (1, 512), 0, 128100),
        jnp.ones((1, 512), dtype=jnp.int32),
        jnp.ones((1, 512), dtype=jnp.int32),
        deterministic=True,
    )

    print(jax.tree_util.tree_map(lambda x: x.shape, params))


if __name__ == "__main__":
    app.run(main)
