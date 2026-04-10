"""Toy language-modeling datasets for small controlled experiments."""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .config import DataConfig, ModelConfig


class SyntheticSequenceDataset(Dataset):
    """A tiny synthetic dataset for next-token prediction experiments."""

    def __init__(
        self,
        size: int,
        model_config: ModelConfig,
        data_config: DataConfig,
        seed: int,
    ) -> None:
        super().__init__()
        self.size = size
        self.model_config = model_config
        self.data_config = data_config
        self.generator = torch.Generator().manual_seed(seed)
        self.sequences = torch.stack([self._make_sequence() for _ in range(size)], dim=0)

    def _make_sequence(self) -> torch.Tensor:
        total_len = self.model_config.block_size + 1
        vocab_size = self.model_config.vocab_size

        if self.data_config.dataset_type == "random":
            return torch.randint(0, vocab_size, (total_len,), generator=self.generator)

        if self.data_config.dataset_type == "repeated_pattern":
            pattern_len = max(1, min(self.data_config.pattern_length, total_len))
            pattern = torch.randint(0, vocab_size, (pattern_len,), generator=self.generator)
            repeats = math.ceil(total_len / pattern_len)
            return pattern.repeat(repeats)[:total_len]

        if self.data_config.dataset_type == "retrieval":
            return self._make_retrieval_sequence(total_len, vocab_size)

        raise ValueError(f"Unsupported dataset_type: {self.data_config.dataset_type}")

    def _make_retrieval_sequence(self, total_len: int, vocab_size: int) -> torch.Tensor:
        pair_count = max(1, min(self.data_config.retrieval_pairs, max(1, vocab_size // 2)))
        key_vocab = max(1, vocab_size // 2)
        value_low = key_vocab if key_vocab < vocab_size else 0

        keys = torch.randperm(key_vocab, generator=self.generator)[:pair_count]
        if value_low == 0:
            values = torch.randint(0, vocab_size, (pair_count,), generator=self.generator)
        else:
            values = torch.randint(value_low, vocab_size, (pair_count,), generator=self.generator)

        memory = torch.stack([keys, values], dim=1).flatten()
        pieces = [memory]
        current_len = memory.numel()

        while current_len < total_len:
            pair_idx = int(torch.randint(0, pair_count, (1,), generator=self.generator))
            pieces.append(keys[pair_idx : pair_idx + 1])
            pieces.append(values[pair_idx : pair_idx + 1])
            current_len += 2

        return torch.cat(pieces)[:total_len]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.sequences[index]
        return sequence[:-1], sequence[1:]


def build_dataloaders(
    model_config: ModelConfig,
    data_config: DataConfig,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 0,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    verbose: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Build deterministic train/validation dataloaders."""

    if data_config.dataset_type == "tinystories":
        from .data_tinystories import build_tinystories_dataloaders

        return build_tinystories_dataloaders(
            model_config=model_config,
            data_config=data_config,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            verbose=verbose,
        )

    train_dataset = SyntheticSequenceDataset(
        size=data_config.train_size,
        model_config=model_config,
        data_config=data_config,
        seed=seed,
    )
    val_dataset = SyntheticSequenceDataset(
        size=data_config.val_size,
        model_config=model_config,
        data_config=data_config,
        seed=seed + 1,
    )

    train_sampler = None
    train_generator = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
    else:
        train_generator = torch.Generator().manual_seed(seed + 2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader
