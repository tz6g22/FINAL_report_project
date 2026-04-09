"""Transformer blocks for the baseline and AttnRes models."""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Tuple

import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .attnres import ActivationOverrideMap, AttnResSite
from .config import ModelConfig
from .history import ResidualHistory
from .mlp import GPTMLP


def _maybe_override(
    key: str,
    tensor: torch.Tensor,
    cache: Dict[str, torch.Tensor] | None,
    activation_overrides: ActivationOverrideMap | None,
) -> torch.Tensor:
    override = None if activation_overrides is None else activation_overrides.get(key)
    if override is not None:
        tensor = override(tensor) if callable(override) else override
    if cache is not None:
        cache[key] = tensor
    return tensor


class StandardGPTBlock(nn.Module):
    """Standard GPT-2 style pre-norm residual block."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = GPTMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cache: Dict[str, torch.Tensor] | None = None,
        activation_overrides: ActivationOverrideMap | None = None,
    ) -> torch.Tensor:
        prefix = f"blocks.{self.layer_idx}"
        x = _maybe_override(f"{prefix}.input", x, cache, activation_overrides)

        attn_outputs = self.attn(self.ln_1(x))
        if cache is not None:
            cache[f"{prefix}.attn_probs"] = attn_outputs["attn_probs"]
            cache[f"{prefix}.q"] = attn_outputs["q"]
            cache[f"{prefix}.k"] = attn_outputs["k"]
            cache[f"{prefix}.v"] = attn_outputs["v"]
        attn_out = _maybe_override(f"{prefix}.attn_out", attn_outputs["output"], cache, activation_overrides)
        x = x + attn_out

        mlp_out = self.mlp(self.ln_2(x))
        mlp_out = _maybe_override(f"{prefix}.mlp_out", mlp_out, cache, activation_overrides)
        x = x + mlp_out
        x = _maybe_override(f"{prefix}.output", x, cache, activation_overrides)
        return x


class AttnResGPTBlock(nn.Module):
    """GPT block where local residual adds are replaced by AttnRes sites."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = GPTMLP(config)
        self.pre_attn_site = AttnResSite(config.n_embd, norm_eps=config.attnres_norm_eps)
        self.pre_mlp_site = AttnResSite(config.n_embd, norm_eps=config.attnres_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        history: ResidualHistory,
        cache: Dict[str, torch.Tensor] | None = None,
        activation_overrides: ActivationOverrideMap | None = None,
    ) -> Tuple[torch.Tensor, ResidualHistory]:
        prefix = f"blocks.{self.layer_idx}"
        x = _maybe_override(f"{prefix}.input", x, cache, activation_overrides)

        pre_attn = self.pre_attn_site(
            history=history,
            current=x,
            cache=cache,
            cache_prefix=f"{prefix}.attnres.pre_attn",
            activation_overrides=activation_overrides,
        )
        history.append(x, site_id=f"{prefix}.pre_attn", layer_id=self.layer_idx)

        attn_outputs = self.attn(self.ln_1(pre_attn["aggregated"]))
        if cache is not None:
            cache[f"{prefix}.attn_probs"] = attn_outputs["attn_probs"]
            cache[f"{prefix}.q"] = attn_outputs["q"]
            cache[f"{prefix}.k"] = attn_outputs["k"]
            cache[f"{prefix}.v"] = attn_outputs["v"]
        attn_out = _maybe_override(f"{prefix}.attn_out", attn_outputs["output"], cache, activation_overrides)

        pre_mlp = self.pre_mlp_site(
            history=history,
            current=attn_out,
            cache=cache,
            cache_prefix=f"{prefix}.attnres.pre_mlp",
            activation_overrides=activation_overrides,
        )
        history.append(attn_out, site_id=f"{prefix}.pre_mlp", layer_id=self.layer_idx)

        mlp_out = self.mlp(self.ln_2(pre_mlp["aggregated"]))
        mlp_out = _maybe_override(f"{prefix}.mlp_out", mlp_out, cache, activation_overrides)
        x_next = _maybe_override(f"{prefix}.output", mlp_out, cache, activation_overrides)
        return x_next, history
