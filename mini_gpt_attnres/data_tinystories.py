"""TinyStories dataset pipeline for full training and evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm may be unavailable in minimal environments
    tqdm = None

from .config import DataConfig, ModelConfig


class TokenBlockDataset(Dataset):
    """Causal-LM blocks from one contiguous token stream."""

    def __init__(self, token_ids: list[int] | torch.Tensor, block_size: int, stride: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.total_len = block_size + 1
        token_count = int(token_ids.numel()) if torch.is_tensor(token_ids) else len(token_ids)
        if token_count < self.total_len:
            raise ValueError(
                f"Not enough tokens ({token_count}) for block_size={block_size}. "
                "Increase train_texts/val_texts or lower block_size."
            )
        if torch.is_tensor(token_ids):
            self.tokens = token_ids.to(dtype=torch.long).view(-1).contiguous()
        else:
            self.tokens = torch.tensor(token_ids, dtype=torch.long)
        max_start = token_count - self.total_len
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
    train_tokens: list[int] | torch.Tensor
    val_tokens: list[int] | torch.Tensor
    vocab_size: int


def _resolve_token_cache_paths(data_config: DataConfig) -> dict[str, Path]:
    base_dir = Path(data_config.token_cache_dir).expanduser()
    if not base_dir.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        base_dir = project_root / base_dir
    payload = {
        "dataset_name": data_config.hf_dataset_name,
        "tokenizer_name": data_config.tokenizer_name,
        "text_field": data_config.text_field,
        "train_texts": data_config.train_texts,
        "val_texts": data_config.val_texts,
    }
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    shared = base_dir / f"token_cache_{signature}.pt"
    return {
        "shared": shared,
        "standard": base_dir / f"token_cache_{signature}_standard.pt",
        "attnres": base_dir / f"token_cache_{signature}_attnres.pt",
    }


def _load_cached_assets(path: Path, data_config: DataConfig, verbose: bool = True) -> TinyStoriesAssets | None:
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as error:
        if verbose:
            print(f"[data] token cache load failed at {path}: {error}. Rebuilding cache.", flush=True)
        return None

    expected = {
        "dataset_name": data_config.hf_dataset_name,
        "tokenizer_name": data_config.tokenizer_name,
        "text_field": data_config.text_field,
        "train_texts": data_config.train_texts,
        "val_texts": data_config.val_texts,
    }
    if payload.get("cache_identity") != expected:
        return None

    train_tokens = payload.get("train_tokens")
    val_tokens = payload.get("val_tokens")
    vocab_size = payload.get("vocab_size")
    if train_tokens is None or val_tokens is None or vocab_size is None:
        return None
    if not torch.is_tensor(train_tokens):
        train_tokens = torch.tensor(train_tokens, dtype=torch.long)
    if not torch.is_tensor(val_tokens):
        val_tokens = torch.tensor(val_tokens, dtype=torch.long)
    if verbose:
        print(f"[data] loaded token cache: {path}", flush=True)
    return TinyStoriesAssets(train_tokens=train_tokens, val_tokens=val_tokens, vocab_size=int(vocab_size))


def _save_cached_assets(path: Path, assets: TinyStoriesAssets, data_config: DataConfig, verbose: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cache_identity = {
        "dataset_name": data_config.hf_dataset_name,
        "tokenizer_name": data_config.tokenizer_name,
        "text_field": data_config.text_field,
        "train_texts": data_config.train_texts,
        "val_texts": data_config.val_texts,
    }
    train_tokens = assets.train_tokens
    val_tokens = assets.val_tokens
    if not torch.is_tensor(train_tokens):
        train_tokens = torch.tensor(train_tokens, dtype=torch.long)
    if not torch.is_tensor(val_tokens):
        val_tokens = torch.tensor(val_tokens, dtype=torch.long)

    payload = {
        "cache_identity": cache_identity,
        "train_tokens": train_tokens.cpu(),
        "val_tokens": val_tokens.cpu(),
        "vocab_size": int(assets.vocab_size),
    }
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    if verbose:
        print(f"[data] saved token cache: {path}", flush=True)


def _replicate_cache_file(source_path: Path, target_path: Path, verbose: bool = True) -> None:
    if source_path == target_path:
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.name}.tmp.copy.{os.getpid()}.{uuid.uuid4().hex}")
    shutil.copyfile(source_path, tmp_path)
    os.replace(tmp_path, target_path)
    if verbose:
        print(f"[data] replicated token cache: {target_path}", flush=True)


def _wait_for_cache_paths(
    candidate_paths: list[Path],
    data_config: DataConfig,
    verbose: bool = True,
    timeout_seconds: float = 7200.0,
    poll_interval_seconds: float = 0.5,
) -> TinyStoriesAssets:
    dedup_paths: list[Path] = []
    seen: set[Path] = set()
    for path in candidate_paths:
        if path in seen:
            continue
        seen.add(path)
        dedup_paths.append(path)

    start = time.monotonic()
    while True:
        for path in dedup_paths:
            cached = _load_cached_assets(path, data_config=data_config, verbose=verbose)
            if cached is not None:
                return cached
        if verbose:
            for path in dedup_paths:
                print(f"[data] waiting for token cache file: {path}", flush=True)
        if time.monotonic() - start > timeout_seconds:
            joined = ", ".join(str(path) for path in dedup_paths)
            raise TimeoutError(f"Timed out waiting for token cache files: {joined}")
        time.sleep(poll_interval_seconds)


def _load_hf_split(dataset_name: str, split: str, verbose: bool = True):
    from datasets import load_dataset

    if verbose:
        print(f"[data] loading dataset={dataset_name} split={split}", flush=True)
    return load_dataset(dataset_name, split=split)


def _encode_texts(
    texts: Iterable[str],
    tokenizer,
    eos_token_id: int,
    desc: str,
    verbose: bool = True,
) -> list[int]:
    token_ids: list[int] = []
    total = len(texts) if hasattr(texts, "__len__") else None
    iterator = texts
    if verbose and tqdm is not None:
        iterator = tqdm(texts, total=total, desc=desc, unit="text", dynamic_ncols=True)
    elif verbose and total is not None:
        print(f"[data] {desc}: 0/{total}", flush=True)

    for idx, text in enumerate(iterator, start=1):
        if not text:
            continue
        # TinyStories samples can exceed tokenizer.model_max_length (e.g., GPT-2's 1024).
        # We later split the contiguous stream into smaller LM blocks, so suppress
        # the tokenizer's long-sequence warning here.
        try:
            encoded = tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=False,
                verbose=False,
            )
        except TypeError:
            encoded = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if not encoded:
            continue
        token_ids.extend(encoded)
        token_ids.append(eos_token_id)
        if verbose and tqdm is None and total is not None and (idx % 1000 == 0 or idx == total):
            print(f"[data] {desc}: {idx}/{total}", flush=True)
    return token_ids


def prepare_tinystories_assets(
    model_config: ModelConfig,
    data_config: DataConfig,
    verbose: bool = True,
    allow_cache_build: bool = True,
) -> TinyStoriesAssets:
    """Load TinyStories and tokenize into train/val token streams."""

    cache_paths = _resolve_token_cache_paths(data_config)
    model_type = str(getattr(model_config, "model_type", "")).lower()
    role_path = cache_paths.get(model_type, cache_paths["shared"])
    local_candidate_paths = [role_path, cache_paths["shared"]]

    if data_config.use_token_cache:
        for path in local_candidate_paths:
            cached = _load_cached_assets(path, data_config=data_config, verbose=verbose)
            if cached is None:
                continue
            if model_config.vocab_size != cached.vocab_size:
                model_config.vocab_size = cached.vocab_size
            if path != role_path:
                _save_cached_assets(role_path, assets=cached, data_config=data_config, verbose=verbose)
            return cached

        should_build = allow_cache_build and model_type != "attnres"
        if not should_build:
            cached_wait = _wait_for_cache_paths(
                candidate_paths=[
                    role_path,
                    cache_paths["shared"],
                    cache_paths["standard"],
                    cache_paths["attnres"],
                ],
                data_config=data_config,
                verbose=verbose,
            )
            if model_config.vocab_size != cached_wait.vocab_size:
                model_config.vocab_size = cached_wait.vocab_size
            if role_path != cache_paths["shared"] and not role_path.exists():
                # Ensure this role gets its own cache replica once a shared/source cache appears.
                _save_cached_assets(role_path, assets=cached_wait, data_config=data_config, verbose=verbose)
            return cached_wait

        if verbose:
            print("[data] token cache miss; building tokenized TinyStories assets...", flush=True)
        assets = _prepare_tinystories_assets_uncached(
            model_config=model_config,
            data_config=data_config,
            verbose=verbose,
        )
        _save_cached_assets(cache_paths["shared"], assets=assets, data_config=data_config, verbose=verbose)
        _replicate_cache_file(cache_paths["shared"], cache_paths["standard"], verbose=verbose)
        _replicate_cache_file(cache_paths["shared"], cache_paths["attnres"], verbose=verbose)
        return assets

    return _prepare_tinystories_assets_uncached(
        model_config=model_config,
        data_config=data_config,
        verbose=verbose,
    )


def _prepare_tinystories_assets_uncached(
    model_config: ModelConfig,
    data_config: DataConfig,
    verbose: bool = True,
) -> TinyStoriesAssets:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(data_config.tokenizer_name, use_fast=True)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError(f"Tokenizer {data_config.tokenizer_name} has no eos_token_id.")

    train_split = "train" if data_config.train_texts is None else f"train[:{data_config.train_texts}]"
    try:
        val_split = "validation" if data_config.val_texts is None else f"validation[:{data_config.val_texts}]"
        val_raw = _load_hf_split(data_config.hf_dataset_name, val_split, verbose=verbose)
    except Exception:
        if data_config.val_texts is None:
            raise RuntimeError(
                "Validation split is unavailable and val_texts is not set. "
                "Provide --val_texts to carve a validation subset from train."
            )
        offset = 0 if data_config.train_texts is None else data_config.train_texts
        val_split = f"train[{offset}:{offset + data_config.val_texts}]"
        val_raw = _load_hf_split(data_config.hf_dataset_name, val_split, verbose=verbose)

    train_raw = _load_hf_split(data_config.hf_dataset_name, train_split, verbose=verbose)
    if data_config.text_field not in train_raw.column_names:
        raise ValueError(
            f"Text field '{data_config.text_field}' not found in dataset columns: {train_raw.column_names}"
        )
    if data_config.text_field not in val_raw.column_names:
        raise ValueError(
            f"Text field '{data_config.text_field}' not found in dataset columns: {val_raw.column_names}"
        )

    train_tokens = _encode_texts(
        train_raw[data_config.text_field],
        tokenizer,
        eos_token_id=eos_token_id,
        desc="tokenizing train split",
        verbose=verbose,
    )
    val_tokens = _encode_texts(
        val_raw[data_config.text_field],
        tokenizer,
        eos_token_id=eos_token_id,
        desc="tokenizing val split",
        verbose=verbose,
    )
    vocab_size = int(tokenizer.vocab_size)
    if tokenizer.eos_token_id is not None:
        vocab_size = max(vocab_size, int(tokenizer.eos_token_id) + 1)

    if model_config.vocab_size != vocab_size:
        model_config.vocab_size = vocab_size

    assets = TinyStoriesAssets(
        train_tokens=torch.tensor(train_tokens, dtype=torch.long),
        val_tokens=torch.tensor(val_tokens, dtype=torch.long),
        vocab_size=vocab_size,
    )
    return assets


def build_tinystories_dataloaders(
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
    """Build TinyStories train/val dataloaders for causal LM."""

    assets = prepare_tinystories_assets(
        model_config=model_config,
        data_config=data_config,
        verbose=verbose,
        allow_cache_build=(not distributed) or (rank == 0),
    )
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
