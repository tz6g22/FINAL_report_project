"""Kimi-style Attention Residuals over residual history sites."""

from __future__ import annotations

from typing import Callable, Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .history import ResidualHistory

ActivationOverrideMap = Mapping[str, torch.Tensor | Callable[[torch.Tensor], torch.Tensor]]


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


class AttnResSite(nn.Module):
    """Static-query attention over residual history plus the current site."""

    def __init__(self, n_embd: int, norm_eps: float = 1e-5) -> None:
        super().__init__()
        self.learned_query = nn.Parameter(torch.randn(n_embd) / n_embd**0.5)
        self.norm_eps = norm_eps

    def forward(
        self,
        history: ResidualHistory,
        current: torch.Tensor,
        cache: Dict[str, torch.Tensor] | None = None,
        cache_prefix: str | None = None,
        activation_overrides: ActivationOverrideMap | None = None,
    ) -> Dict[str, torch.Tensor]:
        if cache_prefix is None:
            raise ValueError("AttnResSite requires a cache_prefix for stable site naming.")

        batch_size, seq_len, hidden_dim = current.shape
        empty_history = current.new_empty(batch_size, seq_len, 0, hidden_dim)
        history_states = history.as_tensor() if len(history) > 0 else empty_history

        current = _maybe_override(
            f"{cache_prefix}.current",
            current,
            cache,
            activation_overrides,
        )
        history_states = _maybe_override(
            f"{cache_prefix}.history_states",
            history_states,
            cache,
            activation_overrides,
        )

        values = torch.cat([history_states, current.unsqueeze(2)], dim=2)
        keys = F.layer_norm(values, (hidden_dim,), eps=self.norm_eps)
        scores = torch.einsum("btsd,d->bts", keys, self.learned_query)
        scores = _maybe_override(
            f"{cache_prefix}.scores",
            scores,
            cache,
            activation_overrides,
        )

        weights = torch.softmax(scores, dim=2)
        weights = _maybe_override(
            f"{cache_prefix}.weights",
            weights,
            cache,
            activation_overrides,
        )

        aggregated = torch.sum(values * weights.unsqueeze(-1), dim=2)
        aggregated = _maybe_override(
            f"{cache_prefix}.aggregated",
            aggregated,
            cache,
            activation_overrides,
        )

        return {
            "current": current,
            "history_states": history_states,
            "scores": scores,
            "weights": weights,
            "aggregated": aggregated,
        }
