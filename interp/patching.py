"""Activation patching helpers."""

from __future__ import annotations

from typing import Callable, Dict

import torch

from .cache import ActivationCache


def patch_from_cache(source_cache: ActivationCache, cache_key: str) -> Dict[str, torch.Tensor]:
    """Patch an activation by replacing it with a cached source activation."""

    return {cache_key: source_cache[cache_key].clone()}


def patch_attention_head_override(
    source_cache: ActivationCache,
    cache_key: str,
    head_index: int,
) -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    """Patch only one head from a source cache into a target activation."""

    source = source_cache[cache_key]

    def apply(target: torch.Tensor) -> torch.Tensor:
        patched = target.clone()
        patched[:, head_index] = source[:, head_index]
        return patched

    return {cache_key: apply}
