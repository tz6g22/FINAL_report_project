"""Ablation helpers that plug into the model activation override interface."""

from __future__ import annotations

from typing import Callable, Dict

import torch


def zero_ablation_override(cache_key: str) -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    """Replace an activation with zeros."""

    return {cache_key: lambda tensor: torch.zeros_like(tensor)}


def mean_ablation_override(
    cache_key: str,
    reference: torch.Tensor,
    reduce_dim: int = 0,
) -> Dict[str, torch.Tensor]:
    """Replace an activation with the mean of a reference tensor."""

    mean_tensor = reference.mean(dim=reduce_dim, keepdim=True).expand_as(reference)
    return {cache_key: mean_tensor}


def zero_attention_head_override(
    cache_key: str,
    head_index: int,
) -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    """Zero out one attention head inside a cached [batch, head, ...] tensor."""

    def apply(tensor: torch.Tensor) -> torch.Tensor:
        ablated = tensor.clone()
        ablated[:, head_index] = 0
        return ablated

    return {cache_key: apply}
