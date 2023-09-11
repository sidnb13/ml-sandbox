from dataclasses import dataclass

from ml_collections import config_dict


@dataclass
class RestConfig:
    pass


@dataclass
class TrainingConfig:
    pass


@dataclass
class DatasetConfig:
    pass


@dataclass
class TransformerConfig:
    embedding_dim: int = 512
    ffn_dim: int = 1024
    num_heads: int = 4
    layers: int = 6
    residual_dropout: float = 0.1
    attention_dropout: float = 0.1
    max_seq_len: int = 512
    vocab_size: int = 32000


def get_config():
    config = config_dict.ConfigDict()

    return config
