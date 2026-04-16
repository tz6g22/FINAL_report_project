#!/usr/bin/env python3
"""Extract activation shards for SAE training from existing model checkpoints."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.data import build_dataloaders
from scripts.evaluate import load_checkpoint
from stream_analysis.path_utils import format_project_path, resolve_project_path
from toygpt2.model import GPTBase

from .config import SAEExtractConfig
from .utils import (
    configure_logging,
    default_activation_dir,
    format_checkpoint_step,
    maybe_tqdm,
    prepare_output_dir,
    resolve_device,
    stage_progress,
    write_json,
)

logger = logging.getLogger("sae.extract")


@dataclass
class ExtractionSummary:
    """Lightweight summary returned after shard extraction."""

    output_dir: str
    meta_path: str
    num_tokens: int
    num_shards: int
    d_model: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "output_dir": self.output_dir,
            "meta_path": self.meta_path,
            "num_tokens": self.num_tokens,
            "num_shards": self.num_shards,
            "d_model": self.d_model,
        }


class BlockInputExtractor:
    """Extract one activation site from one layer using the existing cache path."""

    SUPPORTED_SITES = ("input", "attn_out", "mlp_out", "output", "final_residual")

    def __init__(
        self,
        model: GPTBase,
        *,
        layer_idx: int,
        site: str,
        device: torch.device,
    ) -> None:
        self.model = model
        self.layer_idx = int(layer_idx)
        self.site = str(site).strip().lower()
        self.device = device
        if self.site not in self.SUPPORTED_SITES:
            raise ValueError(f"Unsupported site {site!r}. Expected one of {self.SUPPORTED_SITES}.")
        if self.layer_idx < 0:
            raise ValueError("layer_idx must be non-negative.")

    def extract_site_tensor(self, batch_inputs: torch.Tensor) -> torch.Tensor:
        """Return raw [batch, seq_len, d_model] activations for one input batch."""

        inputs = batch_inputs.to(device=self.device, dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs, return_intermediates=True)
        tensor = self._resolve_site_tensor(outputs)
        if tensor.ndim != 3:
            raise RuntimeError(
                f"Extracted activation tensor must have shape [B, T, D], got {tuple(tensor.shape)}."
            )
        batch_size, seq_len, d_model = tensor.shape
        if batch_size != inputs.shape[0] or seq_len != inputs.shape[1]:
            raise RuntimeError(
                "Extracted activation shape does not align with input batch shape: "
                f"inputs={tuple(inputs.shape)}, activations={tuple(tensor.shape)}."
            )
        return tensor.detach().cpu().to(dtype=torch.float32)

    def extract_batch(self, batch_inputs: torch.Tensor) -> torch.Tensor:
        """Return flattened [num_tokens, d_model] activations for one input batch."""

        tensor = self.extract_site_tensor(batch_inputs)
        batch_size, seq_len, d_model = tensor.shape
        return tensor.reshape(batch_size * seq_len, d_model)

    def _resolve_site_tensor(self, outputs: Mapping[str, object]) -> torch.Tensor:
        intermediates = outputs.get("intermediates")
        if not isinstance(intermediates, Mapping):
            raise RuntimeError("Model outputs must contain a usable 'intermediates' mapping.")

        n_layer = int(getattr(self.model.config, "n_layer", 0))
        n_embd = int(getattr(self.model.config, "n_embd", 0))
        if self.site == "final_residual":
            expected_layer = n_layer - 1
            if self.layer_idx != expected_layer:
                raise ValueError(
                    "site='final_residual' is only defined on the last layer. "
                    f"Expected layer_idx={expected_layer}, got {self.layer_idx}."
                )
            block_key = f"blocks.{expected_layer}.output"
            if block_key not in intermediates:
                raise KeyError(f"Missing required cache key {block_key!r}.")
            tensor = intermediates[block_key]
            if not torch.is_tensor(tensor):
                raise TypeError(f"Cache key {block_key!r} did not produce a tensor.")
            resolved = self.model.ln_f(tensor)
        else:
            key = f"blocks.{self.layer_idx}.{self.site}"
            if key not in intermediates:
                raise KeyError(f"Missing required cache key {key!r}.")
            resolved = intermediates[key]
            if not torch.is_tensor(resolved):
                raise TypeError(f"Cache key {key!r} did not produce a tensor.")

        if resolved.ndim != 3 or resolved.shape[-1] != n_embd:
            raise RuntimeError(
                f"Resolved site tensor has invalid shape {tuple(resolved.shape)}; expected [B, T, {n_embd}]."
            )
        return resolved


def _dataset_name_from_experiment(data_config: object) -> str:
    dataset_type = getattr(data_config, "dataset_type", "unknown")
    if dataset_type == "tinystories":
        return str(getattr(data_config, "hf_dataset_name", dataset_type))
    return str(dataset_type)


def _save_shard(path: Path, tensor: torch.Tensor, save_format: str) -> None:
    if save_format == "pt":
        torch.save({"activations": tensor.cpu(), "num_tokens": int(tensor.shape[0])}, path)
        return
    if save_format == "npy":
        np.save(path, tensor.cpu().numpy())
        return
    raise ValueError(f"Unsupported save_format {save_format!r}.")


def extract_activation_shards(
    config: SAEExtractConfig,
    *,
    show_progress: bool = True,
) -> ExtractionSummary:
    """Extract activation shards for one checkpoint / layer / site combination."""

    checkpoint_path = resolve_project_path(config.checkpoint_path)

    device = resolve_device(config.device)
    with stage_progress(
        f"[{config.model_type}] load checkpoint ({config.dataset_split})",
        enabled=show_progress,
    ):
        model, experiment, checkpoint = load_checkpoint(checkpoint_path, device=device)
    ckpt_model_type = str(checkpoint.get("model_type", getattr(experiment.model, "model_type", "unknown")))
    if ckpt_model_type != config.model_type:
        raise ValueError(
            f"--model-type={config.model_type!r} does not match checkpoint model_type={ckpt_model_type!r}."
        )

    checkpoint_step = checkpoint.get("step")
    output_dir = (
        Path(config.out_dir).expanduser().resolve()
        if config.out_dir
        else default_activation_dir(config.model_type, checkpoint_step, config.layer_idx, config.site)
    )
    with stage_progress(
        f"[{config.model_type}] prepare output ({config.dataset_split})",
        enabled=show_progress,
    ):
        prepare_output_dir(output_dir, overwrite=config.overwrite)

    with stage_progress(
        f"[{config.model_type}] build dataloader ({config.dataset_split})",
        enabled=show_progress,
    ):
        train_loader, val_loader = build_dataloaders(
            model_config=experiment.model,
            data_config=experiment.data,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            seed=experiment.train.seed,
            verbose=True,
        )
    dataloader = train_loader if config.dataset_split == "train" else val_loader

    extractor = BlockInputExtractor(
        model,
        layer_idx=config.layer_idx,
        site=config.site,
        device=device,
    )
    shard_ext = ".pt" if config.save_format == "pt" else ".npy"

    buffered: List[torch.Tensor] = []
    buffered_tokens = 0
    total_tokens = 0
    shard_entries: List[Dict[str, Any]] = []
    d_model: int | None = None

    def flush(force: bool = False) -> None:
        nonlocal buffered, buffered_tokens, d_model
        while buffered_tokens >= config.shard_size_tokens or (force and buffered_tokens > 0):
            merged = torch.cat(buffered, dim=0)
            if force or merged.shape[0] <= config.shard_size_tokens:
                current = merged
                remainder = merged.new_empty((0, merged.shape[1]))
            else:
                current = merged[: config.shard_size_tokens]
                remainder = merged[config.shard_size_tokens :]
            if d_model is None:
                d_model = int(current.shape[1])
            shard_index = len(shard_entries)
            shard_name = f"shard_{shard_index:05d}{shard_ext}"
            shard_path = output_dir / shard_name
            _save_shard(shard_path, current, config.save_format)
            shard_entries.append({"path": shard_name, "num_tokens": int(current.shape[0])})
            buffered = [remainder] if remainder.numel() > 0 else []
            buffered_tokens = int(remainder.shape[0]) if remainder.ndim == 2 else 0
            if not force:
                break

    logger.info(
        "extracting SAE activations | model_type=%s | step=%s | layer=%d | site=%s | split=%s | out_dir=%s",
        config.model_type,
        format_checkpoint_step(checkpoint_step),
        config.layer_idx,
        config.site,
        config.dataset_split,
        output_dir,
    )

    approx_batch_tokens = max(1, config.batch_size * int(getattr(experiment.model, "block_size", 1)))
    estimated_batches = len(dataloader)
    if config.max_tokens is not None:
        estimated_batches = min(estimated_batches, math.ceil(config.max_tokens / approx_batch_tokens))

    batch_iterator = maybe_tqdm(
        enumerate(dataloader),
        desc=f"[{config.model_type}] extract {config.dataset_split}",
        total=estimated_batches,
        enabled=show_progress,
        leave=False,
    )

    for batch_index, (inputs, _) in batch_iterator:
        logger.debug("processing batch %d", batch_index)
        activations = extractor.extract_batch(inputs)
        if config.max_tokens is not None:
            remaining = config.max_tokens - total_tokens
            if remaining <= 0:
                break
            if activations.shape[0] > remaining:
                activations = activations[:remaining]
        if activations.numel() == 0:
            continue

        buffered.append(activations)
        buffered_tokens += int(activations.shape[0])
        total_tokens += int(activations.shape[0])
        if d_model is None:
            d_model = int(activations.shape[1])
        flush(force=False)

        if config.max_tokens is not None and total_tokens >= config.max_tokens:
            break

    flush(force=True)
    if total_tokens <= 0 or d_model is None:
        raise RuntimeError("No activation tokens were extracted; check dataset size and max_tokens.")

    metadata = {
        "format_version": "1.0.0",
        "artifact_type": "sae_activation_shards",
        "model_type": config.model_type,
        "checkpoint_path": format_project_path(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "layer_idx": config.layer_idx,
        "site": config.site,
        "d_model": d_model,
        "num_tokens": total_tokens,
        "dataset_name": _dataset_name_from_experiment(experiment.data),
        "dataset_split": config.dataset_split,
        "preprocessing": {"mode": "none"},
        "save_format": config.save_format,
        "batch_size": config.batch_size,
        "shard_size_tokens": config.shard_size_tokens,
        "extraction_config": config.to_dict(),
        "experiment_model_config": experiment.to_dict()["model"],
        "experiment_data_config": experiment.to_dict()["data"],
        "shards": shard_entries,
    }
    meta_path = write_json(metadata, output_dir / "meta.json")
    logger.info(
        "saved %d activation tokens across %d shards to %s",
        total_tokens,
        len(shard_entries),
        output_dir,
    )
    return ExtractionSummary(
        output_dir=str(output_dir),
        meta_path=str(meta_path),
        num_tokens=total_tokens,
        num_shards=len(shard_entries),
        d_model=d_model,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for activation extraction."""

    parser = argparse.ArgumentParser(
        description="Extract activation shards for SAE training from a model checkpoint."
    )
    parser.add_argument("--model-type", type=str, required=True, help="Checkpoint model_type, e.g. standard or attnres.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the source checkpoint.")
    parser.add_argument("--layer", type=int, required=True, help="Transformer layer index.")
    parser.add_argument("--site", type=str, default="input", help="Activation site to extract. Default: input.")
    parser.add_argument("--dataset-split", type=str, default="train", choices=["train", "val"], help="Dataset split.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Optional token-budget cap.")
    parser.add_argument("--out-dir", type=str, default="", help="Optional custom output directory.")
    parser.add_argument("--batch-size", type=int, default=8, help="Extraction batch size.")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader worker count.")
    parser.add_argument("--shard-size-tokens", type=int, default=131072, help="Tokens per shard.")
    parser.add_argument("--save-format", type=str, default="pt", choices=["pt", "npy"], help="Shard file format.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing shard directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for SAE activation extraction."""

    configure_logging()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        config = SAEExtractConfig(
            model_type=args.model_type,
            checkpoint_path=args.checkpoint,
            layer_idx=args.layer,
            site=args.site,
            dataset_split=args.dataset_split,
            max_tokens=args.max_tokens,
            out_dir=args.out_dir,
            batch_size=args.batch_size,
            device=args.device,
            num_workers=args.num_workers,
            shard_size_tokens=args.shard_size_tokens,
            save_format=args.save_format,
            overwrite=args.overwrite,
        )
        summary = extract_activation_shards(config)
    except Exception as error:
        logger.error("%s", error)
        return 1

    logger.info("extraction finished: %s", summary.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
