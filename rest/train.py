"""An implementation of Reinforced Self-Training (ReST)."""

from functools import partial
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app, flags
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import DatasetConfig, RestConfig, TrainingConfig, TransformerConfig


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
            nn.Dense(self.ffn_dim),
            nn.gelu,
            nn.Dense(self.embedding_dim),
            nn.Dropout(self.residual_dropout),
        )
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
        out = out + self.mlp(out, deterministic=deterministic)
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
            nn.Dense(self.ffn_dim),
            nn.gelu,
            nn.Dense(self.embedding_dim),
            nn.Dropout(self.residual_dropout),
        )
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
        out = out + nn.Sequential(
            nn.Dense(self.ffn_dim),
            nn.gelu,
            nn.Dense(self.embedding_dim),
            nn.Dropout(self.residual_dropout, deterministic=deterministic),
        )(out)
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
        causal_mask = nn.make_causal_mask(jnp.ones(B, input_len))
        emb_tokens = self.embedding(input_ids)
        emb_pos = self.pos_embedding[:, :input_len, :]
        out_ = emb_tokens + emb_pos

        for encoder_block in self.encoder_blocks:
            out_ = encoder_block(out_, attn_mask, deterministic=deterministic)

        out = self.embedding(targets) + self.pos_embedding[:, :target_len, :]

        for decoder_block in self.decoder_blocks:
            out = decoder_block(
                out_, out, attn_mask, causal_mask, deterministic=deterministic
            )

        out = jnp.matmul(out, self.embedding.embedding)  # (B, T, E)
        out = nn.log_softmax(out, axis=-1)
        return out
