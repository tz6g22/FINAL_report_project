"""Lightweight activation cache helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator

import torch


@dataclass
class ActivationCache:
    """A minimal cache wrapper designed for interpretability workflows."""

    data: Dict[str, torch.Tensor]
    metadata: Dict[str, object] = field(default_factory=dict)

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str, default: torch.Tensor | None = None) -> torch.Tensor | None:
        return self.data.get(key, default)

    def keys(self) -> Iterable[str]:
        return self.data.keys()

    def items(self) -> Iterable[tuple[str, torch.Tensor]]:
        return self.data.items()

    def values(self) -> Iterable[torch.Tensor]:
        return self.data.values()

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def to_cpu(self) -> "ActivationCache":
        cpu_data = {key: value.detach().cpu() for key, value in self.data.items()}
        return ActivationCache(cpu_data, metadata=dict(self.metadata))

    def clone(self) -> "ActivationCache":
        cloned = {key: value.clone() for key, value in self.data.items()}
        return ActivationCache(cloned, metadata=dict(self.metadata))

    def subset(self, prefix: str) -> "ActivationCache":
        selected = {key: value for key, value in self.data.items() if key.startswith(prefix)}
        return ActivationCache(selected, metadata=dict(self.metadata))
