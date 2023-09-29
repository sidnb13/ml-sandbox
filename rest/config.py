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
    ds_name: str = "Muennighoff/flan"
    reward_tokenizer: str = "microsoft/deberta-v3-large"
    input_column: str = "inputs"
    target_column: str = "targets"
    max_seq_len: int = 512
    # for DeBERTA model used
    vocab_size: int = 128100


@dataclass
class TransformerConfig:
    embedding_dim: int = 512
    ffn_dim: int = 1024
    num_heads: int = 4
    layers: int = 6
    residual_dropout: float = 0.1
    attention_dropout: float = 0.1
    max_seq_len: int = 512
    # for DeBERTA model used
    vocab_size: int = 128100

def get_config():
    config = config_dict.ConfigDict()

    # TODO

    return config
