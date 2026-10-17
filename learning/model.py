"""
'GPT' model - decoder only architecture
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_embd: int = 256
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
