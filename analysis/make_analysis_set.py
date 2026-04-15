#!/usr/bin/env python3
"""Build a fixed, reproducible analysis set for CKA on residual stream / block inputs.

This script constructs a deterministic subset of token windows from the
TinyStories validation split (or another split).  The output ``.pt`` file
contains everything a downstream ``extract_residuals.py`` needs: input_ids,
labels, target positions, and rich metadata for provenance tracking.

The windows are produced by the **same** ``TokenBlockDataset`` slicing logic
used during training, so every analysis sample is a valid training-format
window.  The token cache system is reused to guarantee identical tokenization.

Usage example
-------------
::

    python analysis/make_analysis_set.py \\
        --checkpoint-config toygpt2_runs/tinystories_dual/standard/config.json \\
        --split val \\
        --num-samples 64 \\
        --seed 1234 \\
        --position-mode last \\
        --output artifacts/analysis_sets/tinystories_val_n64_seed1234.pt

Notes on teacher-forcing alignment
-----------------------------------
For a window of ``block_size + 1`` tokens ``[t_0, t_1, ..., t_{block_size}]``:

- ``input_ids  = [t_0, t_1, ..., t_{block_size-1}]``   (length = block_size)
- ``labels     = [t_1, t_2, ..., t_{block_size}]``      (length = block_size)

When we say "analyse position *t*", we mean position *t* in ``input_ids``.
The model's residual state at position *t* is used to predict ``labels[t]``
(= ``t_{t+1}`` in the original window).  ``position_mode=last`` therefore
sets ``target_positions[i] = block_size - 1`` for every sample.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import torch

# ---------------------------------------------------------------------------
# Ensure project root is importable regardless of how the script is invoked.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from toygpt2.config import DataConfig, ExperimentConfig, ModelConfig
from data.data_tinystories import TokenBlockDataset, prepare_tinystories_assets

# ---------------------------------------------------------------------------
# Version tag embedded in every output file for forward-compatibility checks.
# ---------------------------------------------------------------------------
_SCRIPT_VERSION = "1.0.1"

logger = logging.getLogger("make_analysis_set")


# ===================================================================
# Config loading
# ===================================================================

def load_experiment_config(
    config_path: Optional[str],
    checkpoint_config_path: Optional[str],
) -> tuple[ExperimentConfig, str, str]:
    """Load an ``ExperimentConfig`` from a JSON file.

    ``--checkpoint-config`` takes priority.  If neither is given the
    function raises an error.
    """
    config_source = "checkpoint_config" if checkpoint_config_path is not None else "config"
    path = checkpoint_config_path or config_path
    if path is None:
        raise ValueError(
            "At least one of --config or --checkpoint-config must be provided."
        )
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    logger.info("Loading experiment config from %s", resolved)
    return ExperimentConfig.load_json(resolved), str(resolved), config_source


def canonical_split_name(split: str) -> str:
    """Return the normalized split label used throughout this script."""

    split_lower = split.lower()
    if split_lower in ("val", "validation"):
        return "val"
    if split_lower == "train":
        return "train"
    raise ValueError(
        f"Unsupported split '{split}'. Use 'train', 'val', or 'validation'."
    )


def _resolve_optional_path(path: Optional[str]) -> Optional[str]:
    """Resolve an optional path to an absolute string for metadata."""

    if path is None:
        return None
    return str(Path(path).expanduser().resolve())


def resolve_group_label(
    group_label: Optional[str],
    split: str,
) -> tuple[str, bool]:
    """Return the final group label and whether it was auto-generated."""

    if group_label is not None:
        return group_label, False

    split_label = canonical_split_name(split)
    defaults = {
        "train": "general_train",
        "val": "general_val",
    }
    return defaults[split_label], True


# ===================================================================
# Analysis window construction
# ===================================================================

def build_analysis_windows(
    data_config: DataConfig,
    model_config: ModelConfig,
    split: str,
    num_samples: int,
    seed: int,
    verbose: bool = True,
) -> tuple[torch.Tensor, list[int]]:
    """Return ``(tokens_full, source_indices)``.

    ``tokens_full`` has shape ``(num_samples, block_size + 1)`` — each row is
    a complete teacher-forcing window identical to what
    ``TokenBlockDataset.__getitem__`` would return (concatenated).

    ``source_indices`` records which window index (into the
    ``TokenBlockDataset.starts`` list) each sample came from.

    Parameters
    ----------
    split : str
        ``"val"`` / ``"validation"`` use the validation token stream,
        ``"train"`` uses the training token stream.
    """

    # --- load cached token streams (same path as training) ---------------
    assets = prepare_tinystories_assets(
        model_config=model_config,
        data_config=data_config,
        verbose=verbose,
        allow_cache_build=True,
    )

    # --- pick the right token stream for the requested split -------------
    split_label = canonical_split_name(split)
    if split_label == "val":
        token_stream = assets.val_tokens
    elif split_label == "train":
        token_stream = assets.train_tokens
    else:
        raise AssertionError(f"Unhandled normalized split: {split_label}")

    # --- build the same window index that training uses ------------------
    block_size = model_config.block_size
    stride = data_config.block_stride if data_config.block_stride > 0 else block_size
    dataset = TokenBlockDataset(
        token_ids=token_stream,
        block_size=block_size,
        stride=stride,
    )

    total_windows = len(dataset)
    logger.info(
        "Split=%s | total tokens=%d | block_size=%d | stride=%d | total windows=%d",
        split_label,
        token_stream.numel(),
        block_size,
        stride,
        total_windows,
    )

    if total_windows < num_samples:
        raise ValueError(
            f"Not enough windows ({total_windows}) to sample {num_samples} "
            f"unique analysis samples from split='{split_label}'. "
            "Lower --num-samples or use a larger dataset."
        )

    # --- deterministic sampling ------------------------------------------
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total_windows, generator=rng)
    selected_indices = perm[:num_samples].sort().values.tolist()

    # --- extract full windows (block_size + 1 each) ----------------------
    total_len = block_size + 1
    tokens_full = torch.empty(num_samples, total_len, dtype=torch.long)
    for i, win_idx in enumerate(selected_indices):
        start = dataset.starts[win_idx]
        tokens_full[i] = dataset.tokens[start : start + total_len]

    logger.info("Sampled %d unique windows (seed=%d).", num_samples, seed)
    return tokens_full, selected_indices


# ===================================================================
# Sample ID generation
# ===================================================================

def make_sample_ids(
    dataset_type: str,
    split: str,
    source_indices: list[int],
) -> list[str]:
    """Generate stable, human-readable sample identifiers.

    Format: ``{dataset_type}_{split}_idx{global_window_index:06d}``
    """
    split_label = canonical_split_name(split)
    return [
        f"{dataset_type}_{split_label}_idx{idx:06d}"
        for idx in source_indices
    ]


# ===================================================================
# Target position logic
# ===================================================================

def build_target_positions(
    num_samples: int,
    block_size: int,
    position_mode: str,
    custom_position: Optional[int] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return ``(target_positions, all_positions)``.

    ``target_positions`` has shape ``(num_samples,)`` and holds the default
    analysis position for each sample.

    ``all_positions`` is ``None`` unless ``position_mode="all"``, in which
    case it is ``torch.arange(block_size)``.

    Teacher-forcing alignment reminder (see module docstring):
    position *t* in ``input_ids`` produces a residual state whose prediction
    target is ``labels[t]``.
    """
    if position_mode == "last":
        target_positions = torch.full(
            (num_samples,), block_size - 1, dtype=torch.long,
        )
        all_positions = None

    elif position_mode == "custom":
        if custom_position is None:
            raise ValueError(
                "position_mode='custom' requires --custom-position to be set."
            )
        if not (0 <= custom_position < block_size):
            raise ValueError(
                f"custom_position={custom_position} is out of range "
                f"[0, {block_size})."
            )
        target_positions = torch.full(
            (num_samples,), custom_position, dtype=torch.long,
        )
        all_positions = None

    elif position_mode == "all":
        # Default analysis position is still the last one.
        target_positions = torch.full(
            (num_samples,), block_size - 1, dtype=torch.long,
        )
        all_positions = torch.arange(block_size, dtype=torch.long)

    else:
        raise ValueError(
            f"Unknown position_mode='{position_mode}'. "
            "Choose from: last, all, custom."
        )

    return target_positions, all_positions


# ===================================================================
# Validation
# ===================================================================

def validate_analysis_set(analysis_set: dict, block_size: int) -> None:
    """Run strict shape and consistency checks on the analysis set dict.

    Raises ``ValueError`` on any violation.
    """
    meta = analysis_set["meta"]
    num_samples = meta["num_samples"]

    # --- tokens_full -----------------------------------------------------
    tf = analysis_set["tokens_full"]
    if tf.shape != (num_samples, block_size + 1):
        raise ValueError(
            f"tokens_full shape {tuple(tf.shape)} != "
            f"expected ({num_samples}, {block_size + 1})"
        )

    # --- input_ids / labels ----------------------------------------------
    iids = analysis_set["input_ids"]
    labs = analysis_set["labels"]
    expected_shape = (num_samples, block_size)
    if iids.shape != expected_shape:
        raise ValueError(
            f"input_ids shape {tuple(iids.shape)} != expected {expected_shape}"
        )
    if labs.shape != expected_shape:
        raise ValueError(
            f"labels shape {tuple(labs.shape)} != expected {expected_shape}"
        )

    # --- cross-check: input_ids == tokens_full[:, :-1] -------------------
    if not torch.equal(iids, tf[:, :-1]):
        raise ValueError("input_ids is not equal to tokens_full[:, :-1]")
    if not torch.equal(labs, tf[:, 1:]):
        raise ValueError("labels is not equal to tokens_full[:, 1:]")

    # --- target_positions ------------------------------------------------
    tp = analysis_set["target_positions"]
    if tp.shape != (num_samples,):
        raise ValueError(
            f"target_positions shape {tuple(tp.shape)} != ({num_samples},)"
        )
    if (tp < 0).any() or (tp >= block_size).any():
        raise ValueError(
            f"target_positions has values outside [0, {block_size}): "
            f"min={tp.min().item()}, max={tp.max().item()}"
        )

    # --- sample_ids ------------------------------------------------------
    sids = analysis_set["sample_ids"]
    if len(sids) != num_samples:
        raise ValueError(
            f"sample_ids length {len(sids)} != num_samples {num_samples}"
        )
    if len(set(sids)) != len(sids):
        raise ValueError("sample_ids contains duplicates")

    # --- group_labels ----------------------------------------------------
    gl = analysis_set["group_labels"]
    if len(gl) != num_samples:
        raise ValueError(
            f"group_labels length {len(gl)} != num_samples {num_samples}"
        )

    # --- all_positions (optional) ----------------------------------------
    ap = analysis_set.get("all_positions")
    if ap is not None:
        if ap.shape != (block_size,):
            raise ValueError(
                f"all_positions shape {tuple(ap.shape)} != ({block_size},)"
            )

    logger.info("Validation passed: all shapes and invariants OK.")


# ===================================================================
# Save
# ===================================================================

def save_analysis_set(
    analysis_set: dict,
    output_path: Path,
    overwrite: bool = False,
) -> None:
    """Persist the analysis set to a ``.pt`` file."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}\n"
            "Use --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(analysis_set, output_path)
    logger.info("Saved analysis set to %s", output_path)


# ===================================================================
# CLI
# ===================================================================

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fixed, reproducible analysis set for CKA on "
            "residual stream / block inputs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- config sources --------------------------------------------------
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to an ExperimentConfig JSON file.",
    )
    parser.add_argument(
        "--checkpoint-config",
        type=str,
        default=None,
        help=(
            "Path to a config.json inside a run directory. "
            "Takes priority over --config."
        ),
    )

    # --- data selection --------------------------------------------------
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "validation"],
        help="Which data split to sample from (default: val).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="Number of fixed-length windows to include (default: 64).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducible sampling (default: 1234).",
    )

    # --- output ----------------------------------------------------------
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output .pt file path. If not given, defaults to "
            "artifacts/analysis_sets/<auto_name>.pt"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )

    # --- position mode ---------------------------------------------------
    parser.add_argument(
        "--position-mode",
        type=str,
        default="last",
        choices=["last", "all", "custom"],
        help="How to set the default analysis position (default: last).",
    )
    parser.add_argument(
        "--custom-position",
        type=int,
        default=None,
        help=(
            "Fixed position index when --position-mode=custom. "
            "Must satisfy 0 <= pos < block_size."
        ),
    )

    # --- optional metadata -----------------------------------------------
    parser.add_argument(
        "--include-text",
        action="store_true",
        help=(
            "Attempt to decode and save original texts alongside tokens. "
            "Requires the tokenizer to be available."
        ),
    )
    parser.add_argument(
        "--group-label",
        type=str,
        default=None,
        help=(
            "Group label for the analysis set. If omitted, defaults to "
            "general_val for val/validation and general_train for train."
        ),
    )

    args = parser.parse_args(argv)

    # --- early validation ------------------------------------------------
    if args.config is None and args.checkpoint_config is None:
        parser.error(
            "At least one of --config or --checkpoint-config is required."
        )

    if args.position_mode == "custom" and args.custom_position is None:
        parser.error(
            "--custom-position is required when --position-mode=custom."
        )

    if args.num_samples <= 0:
        parser.error("--num-samples must be a positive integer.")

    return args


# ===================================================================
# Main
# ===================================================================

def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(message)s",
    )

    args = parse_args(argv)

    # ---- 1. Load config -------------------------------------------------
    cfg, resolved_config_path, config_source = load_experiment_config(
        config_path=args.config,
        checkpoint_config_path=args.checkpoint_config,
    )
    model_config: ModelConfig = cfg.model
    data_config: DataConfig = cfg.data
    block_size: int = model_config.block_size
    split_label = canonical_split_name(args.split)
    source_config_path = _resolve_optional_path(args.config)
    source_checkpoint_config_path = _resolve_optional_path(args.checkpoint_config)
    group_label, group_label_was_auto = resolve_group_label(args.group_label, args.split)

    logger.info(
        "Config loaded: dataset_type=%s, tokenizer=%s, block_size=%d, "
        "block_stride=%d",
        data_config.dataset_type,
        data_config.tokenizer_name,
        block_size,
        data_config.block_stride,
    )
    logger.info(
        "Resolved config path: %s (source=%s)",
        resolved_config_path,
        config_source,
    )
    if group_label_was_auto:
        logger.info(
            "Auto-selected group_label=%s for split=%s",
            group_label,
            split_label,
        )
    else:
        logger.info("Using explicit group_label=%s", group_label)

    if data_config.dataset_type != "tinystories":
        raise NotImplementedError(
            f"dataset_type='{data_config.dataset_type}' is not yet supported. "
            "Currently only 'tinystories' is implemented."
        )

    # ---- 2. Build analysis windows --------------------------------------
    tokens_full, source_indices = build_analysis_windows(
        data_config=data_config,
        model_config=model_config,
        split=args.split,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    input_ids = tokens_full[:, :-1].contiguous()
    labels = tokens_full[:, 1:].contiguous()

    # ---- 3. Build sample IDs & group labels -----------------------------
    sample_ids = make_sample_ids(
        dataset_type=data_config.dataset_type,
        split=args.split,
        source_indices=source_indices,
    )
    group_labels = [group_label] * args.num_samples

    # ---- 4. Build target positions --------------------------------------
    target_positions, all_positions = build_target_positions(
        num_samples=args.num_samples,
        block_size=block_size,
        position_mode=args.position_mode,
        custom_position=args.custom_position,
    )

    # ---- 5. Optionally decode texts -------------------------------------
    texts: Optional[List[str]] = None
    if args.include_text:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                data_config.tokenizer_name, use_fast=True,
            )
            # ``texts`` decodes the saved ``input_ids`` rows only, not the
            # full ``tokens_full`` teacher-forcing windows.
            texts = [
                tokenizer.decode(input_ids[i], skip_special_tokens=False)
                for i in range(args.num_samples)
            ]
            logger.info("Decoded texts for %d samples.", args.num_samples)
        except Exception as exc:
            logger.warning(
                "Could not decode texts (tokenizer may be unavailable): %s. "
                "Texts will be saved as None.",
                exc,
            )
            texts = None

    # ---- 6. Assemble output dict ----------------------------------------
    analysis_set: dict = {
        "meta": {
            "dataset_type": data_config.dataset_type,
            "hf_dataset_name": data_config.hf_dataset_name,
            "split": args.split,
            "tokenizer_name": data_config.tokenizer_name,
            "block_size": block_size,
            "block_stride": data_config.block_stride,
            "num_samples": args.num_samples,
            "seed": args.seed,
            "position_mode": args.position_mode,
            "custom_position": args.custom_position,
            "group_label": group_label,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_config_path": source_config_path,
            "source_checkpoint_config_path": source_checkpoint_config_path,
            "resolved_config_path": resolved_config_path,
            "config_source": config_source,
            "texts_semantics": "decoded from input_ids, not tokens_full",
            "script_version": _SCRIPT_VERSION,
        },
        "sample_ids": sample_ids,
        "group_labels": group_labels,
        "tokens_full": tokens_full,
        "input_ids": input_ids,
        "labels": labels,
        "target_positions": target_positions,
        "all_positions": all_positions,
        "source_indices": source_indices,
        "texts": texts,
    }

    # ---- 7. Validate ----------------------------------------------------
    validate_analysis_set(analysis_set, block_size=block_size)

    # ---- 8. Determine output path ---------------------------------------
    if args.output is not None:
        output_path = Path(args.output).expanduser().resolve()
    else:
        auto_name = (
            f"{data_config.dataset_type}_{split_label}"
            f"_n{args.num_samples}_seed{args.seed}.pt"
        )
        output_path = _PROJECT_ROOT / "artifacts" / "analysis_sets" / auto_name

    # ---- 9. Save --------------------------------------------------------
    save_analysis_set(analysis_set, output_path, overwrite=args.overwrite)

    # ---- 10. Print summary ----------------------------------------------
    example_pos = target_positions[0].item()
    print(flush=True)
    print(f"[analysis_set] saved to: {output_path}", flush=True)
    print(f"[analysis_set] num_samples: {args.num_samples}", flush=True)
    print(f"[analysis_set] block_size: {block_size}", flush=True)
    print(f"[analysis_set] position_mode: {args.position_mode}", flush=True)
    print(f"[analysis_set] target_position_example: {example_pos}", flush=True)
    print(
        f"[analysis_set] input_ids shape: {tuple(input_ids.shape)}",
        flush=True,
    )
    print(
        f"[analysis_set] tokens_full shape: {tuple(tokens_full.shape)}",
        flush=True,
    )
    print(
        f"[analysis_set] sample_id[0]: {sample_ids[0]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
