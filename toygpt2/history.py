"""Structured residual history for AttnRes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class ResidualHistoryEntry:
    """One residual-site entry with explicit metadata."""

    state: torch.Tensor
    site_id: str
    layer_id: int


class ResidualHistory:
    """Container for depth/site-wise AttnRes history.

    States are stored as explicit entries rather than a bare list so the caller
    can inspect, ablate, or patch specific residual sites by name.
    """

    def __init__(self) -> None:
        self.states: List[torch.Tensor] = []
        self.site_ids: List[str] = []
        self.layer_ids: List[int] = []

    def append(self, state: torch.Tensor, site_id: str, layer_id: int) -> "ResidualHistory":
        if state.ndim != 3:
            raise ValueError("ResidualHistory expects states shaped [batch, time, hidden].")
        self.states.append(state)
        self.site_ids.append(site_id)
        self.layer_ids.append(layer_id)
        return self

    def as_tensor(self) -> torch.Tensor:
        if not self.states:
            raise ValueError("Cannot stack an empty ResidualHistory.")
        return torch.stack(self.states, dim=2)

    def metadata(self, index: int | None = None) -> Dict[str, int | str] | List[Dict[str, int | str]]:
        if index is None:
            return [
                {"site_id": site_id, "layer_id": layer_id}
                for site_id, layer_id in zip(self.site_ids, self.layer_ids)
            ]
        return {"site_id": self.site_ids[index], "layer_id": self.layer_ids[index]}

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> ResidualHistoryEntry:
        return ResidualHistoryEntry(
            state=self.states[index],
            site_id=self.site_ids[index],
            layer_id=self.layer_ids[index],
        )

