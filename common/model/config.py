"""Config for the decoder-only Transformer LM (common/model/transformer.py).

Shared code path for both Model H (Hindi) and Model L (Nepali): each
language instantiates its own GPT from its own GPTConfig, its own tokenizer
/ vocab, and its own weights. Nothing is shared across languages at runtime.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union

import yaml


@dataclass
class GPTConfig:
    vocab_size: int
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    ffn_dim: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1
    tie_embeddings: bool = True
    pad_id: int = 0

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "GPTConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    def to_dict(self) -> dict:
        return asdict(self)
