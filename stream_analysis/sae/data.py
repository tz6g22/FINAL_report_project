"""Activation-shard datasets and preprocessing helpers for SAE training."""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from .config import resolve_preprocessing_mode
from .utils import read_json


def preprocess_activations(activations: torch.Tensor, mode: str = "none") -> torch.Tensor:
    """Apply deterministic per-sample preprocessing to activation rows."""

    resolved = resolve_preprocessing_mode(mode)
    outputs = activations.to(dtype=torch.float32)
    if "mean-center" in resolved:
        outputs = outputs - outputs.mean(dim=-1, keepdim=True)
    if "unit-norm" in resolved:
        norms = outputs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        outputs = outputs / norms
    return outputs


class ActivationShardDataset(Dataset[torch.Tensor]):
    """Map-style dataset over one activation shard directory."""

    def __init__(
        self,
        activation_dir: str | Path,
        *,
        preprocessing: str = "none",
        mmap: bool = False,
    ) -> None:
        super().__init__()
        self.activation_dir = Path(activation_dir).expanduser().resolve()
        meta_path = self.activation_dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Activation shard directory is missing meta.json: {meta_path}")

        metadata = read_json(meta_path)
        if not isinstance(metadata, Mapping):
            raise TypeError(f"meta.json must contain a JSON object, got {type(metadata).__name__}.")
        shard_entries = metadata.get("shards")
        if not isinstance(shard_entries, Sequence) or not shard_entries:
            raise ValueError("meta.json must contain a non-empty 'shards' list.")

        d_model = metadata.get("d_model")
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError(f"meta.json must contain a positive integer d_model, got {d_model!r}.")

        self.metadata: Dict[str, Any] = dict(metadata)
        self.preprocessing = resolve_preprocessing_mode(preprocessing)
        self.mmap = bool(mmap)
        self.d_in = int(d_model)
        self.shards: List[Dict[str, Any]] = []
        self._cumulative_lengths: List[int] = []
        self._loaded_shard_index: int | None = None
        self._loaded_shard_tensor: torch.Tensor | None = None

        total = 0
        for entry in shard_entries:
            if not isinstance(entry, Mapping):
                raise TypeError("Each shard entry in meta.json must be a mapping.")
            rel_path = entry.get("path")
            num_tokens = entry.get("num_tokens")
            if not isinstance(rel_path, str) or not rel_path:
                raise ValueError(f"Invalid shard path entry: {entry!r}")
            if not isinstance(num_tokens, int) or num_tokens <= 0:
                raise ValueError(f"Invalid shard length entry: {entry!r}")
            shard_path = self.activation_dir / rel_path
            self.shards.append({"path": shard_path, "num_tokens": num_tokens})
            total += num_tokens
            self._cumulative_lengths.append(total)

        declared_total = self.metadata.get("num_tokens")
        if isinstance(declared_total, int) and declared_total != total:
            raise ValueError(
                f"meta.json num_tokens={declared_total} does not match summed shard tokens={total}."
            )
        self._total_tokens = total

    def __len__(self) -> int:
        return self._total_tokens

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} is out of range for dataset of length {len(self)}.")

        shard_idx = bisect.bisect_right(self._cumulative_lengths, index)
        shard_start = 0 if shard_idx == 0 else self._cumulative_lengths[shard_idx - 1]
        local_index = index - shard_start
        shard_tensor = self._load_shard(shard_idx)
        sample = shard_tensor[local_index]
        if sample.ndim != 1 or sample.shape[0] != self.d_in:
            raise RuntimeError(
                f"Shard sample must have shape [{self.d_in}], got {tuple(sample.shape)} from shard {shard_idx}."
            )
        return preprocess_activations(sample.unsqueeze(0), mode=self.preprocessing).squeeze(0)

    def _load_shard(self, shard_index: int) -> torch.Tensor:
        if self._loaded_shard_index == shard_index and self._loaded_shard_tensor is not None:
            return self._loaded_shard_tensor

        entry = self.shards[shard_index]
        path = Path(entry["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Activation shard is missing: {path}")

        if path.suffix == ".pt":
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if torch.is_tensor(payload):
                tensor = payload
            elif isinstance(payload, Mapping) and torch.is_tensor(payload.get("activations")):
                tensor = payload["activations"]
            else:
                raise TypeError(
                    f"Unsupported .pt shard payload type in {path}: {type(payload).__name__}."
                )
        elif path.suffix == ".npy":
            array = np.load(path, mmap_mode="r" if self.mmap else None)
            tensor = torch.from_numpy(array)
        else:
            raise ValueError(f"Unsupported shard suffix {path.suffix!r} for {path}.")

        if tensor.ndim != 2 or tensor.shape[1] != self.d_in:
            raise ValueError(
                f"Shard {path} must have shape [N, {self.d_in}], got {tuple(tensor.shape)}."
            )
        if tensor.shape[0] != int(entry["num_tokens"]):
            raise ValueError(
                f"Shard {path} declares {entry['num_tokens']} tokens but contains {tensor.shape[0]} rows."
            )

        loaded = tensor.detach().cpu().to(dtype=torch.float32).contiguous()
        self._loaded_shard_index = shard_index
        self._loaded_shard_tensor = loaded
        return loaded


def build_activation_dataloaders(
    activation_dir: str | Path,
    *,
    batch_size: int,
    val_fraction: float,
    seed: int,
    preprocessing: str = "none",
    num_workers: int = 0,
    mmap: bool = False,
) -> tuple[DataLoader, DataLoader, Dataset[torch.Tensor], Dataset[torch.Tensor]]:
    """Build deterministic train / val loaders over activation shards."""

    dataset = ActivationShardDataset(activation_dir, preprocessing=preprocessing, mmap=mmap)
    if len(dataset) == 0:
        raise RuntimeError("Activation dataset is empty; cannot build dataloaders.")

    if len(dataset) < 2 or val_fraction <= 0.0:
        train_dataset: Dataset[torch.Tensor] = dataset
        val_dataset: Dataset[torch.Tensor] = dataset
    else:
        val_size = max(1, int(round(len(dataset) * val_fraction)))
        train_size = len(dataset) - val_size
        if train_size <= 0:
            train_size = len(dataset) - 1
            val_size = 1
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader, train_dataset, val_dataset
