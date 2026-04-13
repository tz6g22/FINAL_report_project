"""Feed-forward sublayer used inside GPT blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig


class GPTMLP(nn.Module):
    """A minimal GPT-2 style MLP block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden_dim = int(config.n_embd * config.mlp_ratio)
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.activation = nn.GELU()
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        return self.dropout(x)
