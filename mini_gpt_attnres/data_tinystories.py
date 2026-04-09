"""TinyStories dataset pipeline for minimal smoke training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig, ModelConfig


class TokenBlockDataset(Dataset):
    """Causal-LM blocks from one contiguous token stream."""

    def __init__(self, token_ids: list[int], block_size: int, stride: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.total_len = block_size + 1
        if len(token_ids) < self.total_len:
            raise ValueError(
                f"Not enough tokens ({len(token_ids)}) for block_size={block_size}. "
                "Increase train_texts/val_texts or lower block_size."
            )
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        max_start = len(token_ids) - self.total_len
        self.starts = list(range(0, max_start + 1, max(1, stride)))
        if not self.starts:
            self.starts = [0]

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[index]
        chunk = self.tokens[start : start + self.total_len]
        return chunk[:-1], chunk[1:]


@dataclass
class TinyStoriesAssets:
    train_tokens: list[int]
    val_tokens: list[int]
    vocab_size: int


def _load_hf_split(dataset_name: str, split: str):
    from datasets import load_dataset

    return load_dataset(dataset_name, split=split)


def _encode_texts(texts: Iterable[str], tokenizer, eos_token_id: int) -> list[int]:
    token_ids: list[int] = []
    for text in texts:
        if not text:
            continue
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if not encoded:
            continue
        token_ids.extend(encoded)
        token_ids.append(eos_token_id)
    return token_ids


def prepare_tinystories_assets(model_config: ModelConfig, data_config: DataConfig) -> TinyStoriesAssets:
    """Load TinyStories subset and tokenize into train/val token streams."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(data_config.tokenizer_name, use_fast=True)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError(f"Tokenizer {data_config.tokenizer_name} has no eos_token_id.")

    train_split = f"train[:{data_config.train_texts}]"
    try:
        val_split = f"validation[:{data_config.val_texts}]"
        val_raw = _load_hf_split(data_config.hf_dataset_name, val_split)
    except Exception:
        offset = data_config.train_texts
        val_split = f"train[{offset}:{offset + data_config.val_texts}]"
        val_raw = _load_hf_split(data_config.hf_dataset_name, val_split)

    train_raw = _load_hf_split(data_config.hf_dataset_name, train_split)
    if data_config.text_field not in train_raw.column_names:
        raise ValueError(
            f"Text field '{data_config.text_field}' not found in dataset columns: {train_raw.column_names}"
        )
    if data_config.text_field not in val_raw.column_names:
        raise ValueError(
            f"Text field '{data_config.text_field}' not found in dataset columns: {val_raw.column_names}"
        )

    train_tokens = _encode_texts(train_raw[data_config.text_field], tokenizer, eos_token_id=eos_token_id)
    val_tokens = _encode_texts(val_raw[data_config.text_field], tokenizer, eos_token_id=eos_token_id)
    vocab_size = int(tokenizer.vocab_size)
    if tokenizer.eos_token_id is not None:
        vocab_size = max(vocab_size, int(tokenizer.eos_token_id) + 1)

    if model_config.vocab_size != vocab_size:
        model_config.vocab_size = vocab_size

    return TinyStoriesAssets(
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        vocab_size=vocab_size,
    )


def build_tinystories_dataloaders(
    model_config: ModelConfig,
    data_config: DataConfig,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Build TinyStories train/val dataloaders for causal LM."""

    assets = prepare_tinystories_assets(model_config=model_config, data_config=data_config)
    stride = data_config.block_stride if data_config.block_stride > 0 else model_config.block_size

    train_dataset = TokenBlockDataset(
        token_ids=assets.train_tokens,
        block_size=model_config.block_size,
        stride=stride,
    )
    val_dataset = TokenBlockDataset(
        token_ids=assets.val_tokens,
        block_size=model_config.block_size,
        stride=stride,
    )

    train_generator = torch.Generator().manual_seed(seed + 2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
