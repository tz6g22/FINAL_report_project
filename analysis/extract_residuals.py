#!/usr/bin/env python3
"""Extract residual-stream / block-input states from a saved checkpoint.

This script reads a pre-built analysis set (from ``analysis/make_analysis_set.py``),
loads a saved model checkpoint, runs batched forward passes with
``return_intermediates=True``, and saves residual-stream states into a single
``.pt`` artifact for downstream CKA analysis.

It only does extraction:
- load analysis set
- load checkpoint + config metadata
- run forward passes
- extract ``resid_0`` ... ``resid_{L-1}`` and ``resid_final``
- save a standardized artifact

It does not do:
- CKA computation
- probe training
- plotting
- training-logic changes

Usage example
-------------
::

    python analysis/extract_residuals.py \\
        --checkpoint toygpt2_runs/tinystories_dual/standard/ckpt_standard_last.pt \\
        --config toygpt2_runs/tinystories_dual/standard/config.json \\
        --analysis-set artifacts/analysis_sets/tinystories_val_n64_seed1234.pt \\
        --output artifacts/activations/standard_last_target.pt \\
        --device cuda \\
        --batch-size 32 \\
        --extract-mode target_only

State definitions
-----------------
- ``resid_0`` ... ``resid_{L-1}``: input to transformer block ``i``
  (cache key ``blocks.{i}.input``). ``resid_0`` is validated to match
  ``embedding_out`` when that cache key is available.
- ``resid_final``: output of the last block passed through ``model.ln_f``.
  This is exactly the representation consumed by ``lm_head``.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import copy
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Ensure project root is importable regardless of invocation directory.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.evaluate import load_checkpoint, resolve_device
from toygpt2.config import ExperimentConfig
from toygpt2.model import GPTBase

logger = logging.getLogger("extract_residuals")

_SCRIPT_VERSION = "1.1.1"
_REQUIRED_ANALYSIS_SET_KEYS = {"input_ids", "sample_ids", "group_labels", "target_positions"}
_REQUIRED_CHECKPOINT_KEYS = {
    "step",
    "model_type",
    "model_config",
    "data_config",
    "train_config",
    "model_state",
}
_TARGET_STATE_COLLECTION = "states"
_BOTH_TARGET_COLLECTION = "states_target"
_BOTH_FULL_COLLECTION = "states_full"


StateDict = Dict[str, torch.Tensor]
StateCollections = Dict[str, StateDict]


class AnalysisTensorDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Tensor-backed dataset used only for ordered batched extraction."""

    def __init__(self, input_ids: torch.Tensor, target_positions: torch.Tensor) -> None:
        self.input_ids = input_ids
        self.target_positions = target_positions

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[index], self.target_positions[index]


def _resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _find_config_near_checkpoint(checkpoint_path: Path) -> Optional[Path]:
    """Try to locate ``config.json`` next to, or one directory above, a checkpoint."""

    candidates = [
        checkpoint_path.parent / "config.json",
        checkpoint_path.parent.parent / "config.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _expected_state_names(n_layer: int) -> List[str]:
    if n_layer <= 0:
        raise ValueError(f"n_layer must be positive, got {n_layer}.")
    return [*(f"resid_{idx}" for idx in range(n_layer)), "resid_final"]


def _state_sort_key(name: str) -> tuple[int, str]:
    if name == "resid_final":
        return (10**9, name)
    if name.startswith("resid_"):
        try:
            return (int(name.split("_", maxsplit=1)[1]), name)
        except ValueError:
            return (10**9 + 1, name)
    return (10**9 + 2, name)


def _validate_string_sequence(values: Sequence[object], name: str, num_samples: int) -> None:
    if len(values) != num_samples:
        raise ValueError(f"{name} length {len(values)} != num_samples {num_samples}.")
    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise TypeError(f"{name}[{index}] must be a string, got {type(value).__name__}.")


def load_analysis_set(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate an analysis-set artifact."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Analysis set not found: {resolved}")

    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Analysis set must be a mapping, got {type(payload).__name__}.")

    missing = _REQUIRED_ANALYSIS_SET_KEYS - set(payload.keys())
    if missing:
        raise ValueError(
            f"Analysis set is missing required keys: {sorted(missing)}. "
            f"Available keys: {sorted(payload.keys())}"
        )

    input_ids = payload["input_ids"]
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("analysis_set['input_ids'] must be a torch.Tensor.")
    if input_ids.ndim != 2:
        raise ValueError(
            f"analysis_set['input_ids'] must have shape [N, T], got {tuple(input_ids.shape)}."
        )
    if torch.is_floating_point(input_ids):
        raise TypeError("analysis_set['input_ids'] must contain integer token ids.")
    input_ids = input_ids.to(torch.long)
    payload["input_ids"] = input_ids
    num_samples, seq_len = (int(input_ids.shape[0]), int(input_ids.shape[1]))

    sample_ids = payload["sample_ids"]
    group_labels = payload["group_labels"]
    if not isinstance(sample_ids, Sequence) or isinstance(sample_ids, (str, bytes)):
        raise TypeError("analysis_set['sample_ids'] must be a sequence of strings.")
    if not isinstance(group_labels, Sequence) or isinstance(group_labels, (str, bytes)):
        raise TypeError("analysis_set['group_labels'] must be a sequence of strings.")
    _validate_string_sequence(sample_ids, name="sample_ids", num_samples=num_samples)
    _validate_string_sequence(group_labels, name="group_labels", num_samples=num_samples)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("analysis_set['sample_ids'] contains duplicates.")

    target_positions = payload["target_positions"]
    if not isinstance(target_positions, torch.Tensor):
        raise TypeError("analysis_set['target_positions'] must be a torch.Tensor.")
    if target_positions.ndim != 1 or int(target_positions.shape[0]) != num_samples:
        raise ValueError(
            "analysis_set['target_positions'] must have shape [N]. "
            f"Got {tuple(target_positions.shape)} for num_samples={num_samples}."
        )
    if torch.is_floating_point(target_positions):
        raise TypeError("analysis_set['target_positions'] must contain integer indices.")
    target_positions = target_positions.to(torch.long)
    payload["target_positions"] = target_positions
    if (target_positions < 0).any() or (target_positions >= seq_len).any():
        raise ValueError(
            f"analysis_set['target_positions'] must lie in [0, {seq_len}). "
            f"Observed min={int(target_positions.min().item())}, "
            f"max={int(target_positions.max().item())}."
        )

    labels = payload.get("labels")
    if labels is not None:
        if not isinstance(labels, torch.Tensor):
            raise TypeError("analysis_set['labels'] must be a torch.Tensor when present.")
        if labels.shape != input_ids.shape:
            raise ValueError(
                "analysis_set['labels'] must match input_ids shape. "
                f"Got labels={tuple(labels.shape)}, input_ids={tuple(input_ids.shape)}."
            )
        if torch.is_floating_point(labels):
            raise TypeError("analysis_set['labels'] must contain integer token ids.")
        payload["labels"] = labels.to(torch.long)

    meta = payload.get("meta")
    if meta is not None:
        if not isinstance(meta, Mapping):
            raise TypeError("analysis_set['meta'] must be a mapping when present.")
        if "num_samples" in meta and int(meta["num_samples"]) != num_samples:
            raise ValueError(
                f"analysis_set meta num_samples={meta['num_samples']} does not match "
                f"input_ids.shape[0]={num_samples}."
            )
        if "block_size" in meta and int(meta["block_size"]) != seq_len:
            raise ValueError(
                f"analysis_set meta block_size={meta['block_size']} does not match "
                f"input_ids.shape[1]={seq_len}."
            )

    return dict(payload)


def _compare_model_configs(
    checkpoint_experiment: ExperimentConfig,
    config_experiment: ExperimentConfig,
) -> Dict[str, tuple[object, object]]:
    """Return a field -> (config_json_value, checkpoint_value) mismatch mapping."""

    checkpoint_model = asdict(checkpoint_experiment.model)
    config_model = asdict(config_experiment.model)
    mismatches: Dict[str, tuple[object, object]] = {}
    for key in sorted(checkpoint_model):
        if config_model.get(key) != checkpoint_model.get(key):
            mismatches[key] = (config_model.get(key), checkpoint_model.get(key))
    return mismatches


def load_model_and_config(
    checkpoint_path: str | Path,
    config_path: Optional[str | Path],
    device: torch.device,
) -> tuple[GPTBase, ExperimentConfig, Dict[str, Any]]:
    """Load model/checkpoint metadata while reusing the project's restore logic.

    ``scripts.evaluate.load_checkpoint`` remains the source of truth for model
    reconstruction. An explicit or auto-detected ``config.json`` is only used for
    validation and metadata.
    """

    ckpt_resolved = Path(checkpoint_path).expanduser().resolve()
    if not ckpt_resolved.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_resolved}")

    try:
        model, checkpoint_experiment, checkpoint = load_checkpoint(ckpt_resolved, device=device)
    except KeyError as error:
        raise ValueError(
            "Checkpoint is missing required fields for the scripts/train.py schema. "
            f"Underlying error: {error!s}"
        ) from error

    if not isinstance(model, GPTBase):
        raise TypeError(f"Expected GPTBase model, got {type(model).__name__}.")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint payload must be a mapping, got {type(checkpoint).__name__}.")

    missing = _REQUIRED_CHECKPOINT_KEYS - set(checkpoint.keys())
    if missing:
        raise ValueError(
            f"Checkpoint is missing required keys: {sorted(missing)}. "
            f"Available keys: {sorted(checkpoint.keys())}"
        )

    model.eval()
    checkpoint_model_type = str(checkpoint["model_type"])
    if checkpoint_model_type not in {"standard", "attnres"}:
        raise ValueError(
            f"Unsupported checkpoint model_type='{checkpoint_model_type}'. "
            "Expected 'standard' or 'attnres'."
        )
    if model.config.model_type != checkpoint_model_type:
        raise ValueError(
            f"Restored model_type mismatch: model.config.model_type={model.config.model_type!r}, "
            f"checkpoint['model_type']={checkpoint_model_type!r}."
        )
    if len(model.blocks) != checkpoint_experiment.model.n_layer:
        raise ValueError(
            f"Model block count {len(model.blocks)} does not match config n_layer "
            f"{checkpoint_experiment.model.n_layer}."
        )

    resolved_config_path: Optional[Path] = None
    config_source = "checkpoint_embedded"
    config_validation_status = "embedded_only"
    if config_path is not None:
        resolved_config_path = Path(config_path).expanduser().resolve()
        if not resolved_config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {resolved_config_path}")
        config_source = "explicit"
    else:
        resolved_config_path = _find_config_near_checkpoint(ckpt_resolved)
        if resolved_config_path is not None:
            config_source = "auto_detected"

    if resolved_config_path is not None:
        external_experiment = ExperimentConfig.load_json(resolved_config_path)
        mismatches = _compare_model_configs(checkpoint_experiment, external_experiment)
        if mismatches:
            mismatch_lines = ", ".join(
                f"{field}: config_json={config_value!r}, checkpoint={ckpt_value!r}"
                for field, (config_value, ckpt_value) in sorted(mismatches.items())
            )
            if config_path is not None:
                raise ValueError(
                    "Explicit config.json does not match checkpoint model_config. "
                    f"Mismatched fields: {mismatch_lines}"
                )
            logger.warning(
                "Auto-detected config.json differs from checkpoint model_config and will be ignored. "
                "Mismatched fields: %s",
                mismatch_lines,
            )
            config_validation_status = "auto_detected_mismatch_ignored"
        else:
            config_validation_status = "matched_checkpoint"

    meta = {
        "checkpoint_path": str(ckpt_resolved),
        "config_path": None if resolved_config_path is None else str(resolved_config_path),
        "config_source": config_source,
        "config_validation_status": config_validation_status,
        "model_type": checkpoint_model_type,
        "step": checkpoint.get("step"),
        "n_layer": int(checkpoint_experiment.model.n_layer),
        "n_embd": int(checkpoint_experiment.model.n_embd),
        "block_size": int(checkpoint_experiment.model.block_size),
        "model_config": copy.deepcopy(checkpoint["model_config"]),
    }
    return model, checkpoint_experiment, meta


def _select_target_positions(states: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
    """Select ``states[b, target_positions[b], :]`` for each sample ``b``."""

    batch_size = int(states.shape[0])
    return states[torch.arange(batch_size, device=states.device), target_positions]


def _prepare_for_saving(tensor: torch.Tensor, save_dtype: torch.dtype) -> torch.Tensor:
    return tensor.detach().cpu().to(save_dtype)


def _validate_resid_final_logits_match(
    recomputed_logits: torch.Tensor,
    logits: torch.Tensor,
) -> None:
    """Validate that ``resid_final`` is the tensor consumed by ``lm_head``.

    The comparison is performed in float32 for better stability across
    float16/bfloat16 execution modes. Tiny residual numerical discrepancies can
    arise from precision effects, so we warn on very small mismatches and only
    hard-fail on materially larger deviations.
    """

    recomputed_logits_f32 = recomputed_logits.detach().to(torch.float32)
    logits_f32 = logits.detach().to(torch.float32)
    if not torch.isfinite(recomputed_logits_f32).all():
        raise RuntimeError("Recomputed logits from resid_final contain NaN or Inf values.")
    if not torch.isfinite(logits_f32).all():
        raise RuntimeError("Model outputs['logits'] contain NaN or Inf values.")

    if torch.allclose(recomputed_logits_f32, logits_f32, atol=1e-4, rtol=1e-4):
        return

    abs_diff = (recomputed_logits_f32 - logits_f32).abs()
    rel_diff = abs_diff / logits_f32.abs().clamp_min(1e-6)
    max_abs_diff = float(abs_diff.max().item())
    max_rel_diff = float(rel_diff.max().item())
    if max_abs_diff <= 5e-4 and max_rel_diff <= 5e-3:
        logger.warning(
            "resid_final reproduced logits only up to a very small float32 mismatch "
            "(max_abs_diff=%.6g, max_rel_diff=%.6g). Continuing.",
            max_abs_diff,
            max_rel_diff,
        )
        return

    raise RuntimeError(
        "resid_final did not reproduce model logits through lm_head in float32. "
        f"max_abs_diff={max_abs_diff:.6g}, max_rel_diff={max_rel_diff:.6g}. "
        "The post-ln_f pre-lm_head assumption appears to be violated."
    )


def _collection_names_for_mode(extract_mode: str) -> List[str]:
    """Return the expected saved state collections for an extract mode."""

    if extract_mode == "both":
        return [_BOTH_TARGET_COLLECTION, _BOTH_FULL_COLLECTION]
    if extract_mode in {"target_only", "full_sequence"}:
        return [_TARGET_STATE_COLLECTION]
    raise ValueError(f"Unsupported extract_mode={extract_mode!r}.")


def _expected_state_shape(
    *,
    collection_name: str,
    extract_mode: str,
    num_samples: int,
    seq_len: int,
    n_embd: int,
) -> tuple[int, ...]:
    """Return the expected tensor shape for a saved state collection."""

    if collection_name == _BOTH_TARGET_COLLECTION:
        return (num_samples, n_embd)
    if collection_name == _BOTH_FULL_COLLECTION:
        return (num_samples, seq_len, n_embd)
    if collection_name != _TARGET_STATE_COLLECTION:
        raise ValueError(f"Unknown state collection {collection_name!r}.")
    if extract_mode == "target_only":
        return (num_samples, n_embd)
    if extract_mode == "full_sequence":
        return (num_samples, seq_len, n_embd)
    raise ValueError(
        f"Collection {_TARGET_STATE_COLLECTION!r} is ambiguous for extract_mode={extract_mode!r}."
    )


@torch.no_grad()
def extract_states_for_batch(
    model: GPTBase,
    input_ids: torch.Tensor,
    extract_mode: str,
    target_positions: Optional[torch.Tensor],
    save_dtype: torch.dtype,
) -> StateCollections:
    """Run one forward pass and extract residual-stream states for a batch."""

    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}.")
    if extract_mode in {"target_only", "both"} and target_positions is None:
        raise ValueError(f"target_positions is required for extract_mode={extract_mode!r}.")

    batch_size, seq_len = (int(input_ids.shape[0]), int(input_ids.shape[1]))
    n_layer = int(model.config.n_layer)
    n_embd = int(model.config.n_embd)
    expected_states = _expected_state_names(n_layer)

    outputs = model(input_ids, return_intermediates=True)
    intermediates = outputs.get("intermediates")
    if not isinstance(intermediates, Mapping):
        raise RuntimeError("Model outputs do not contain a usable 'intermediates' mapping.")

    raw_states: StateDict = {}
    for layer_idx in range(n_layer):
        cache_key = f"blocks.{layer_idx}.input"
        if cache_key not in intermediates:
            raise RuntimeError(
                f"Missing expected cache key {cache_key!r}. "
                f"Available keys: {sorted(intermediates.keys())}"
            )
        tensor = intermediates[cache_key]
        if tensor.shape != (batch_size, seq_len, n_embd):
            raise RuntimeError(
                f"Cache key {cache_key!r} has shape {tuple(tensor.shape)}; "
                f"expected {(batch_size, seq_len, n_embd)}."
            )
        raw_states[f"resid_{layer_idx}"] = tensor

    embedding_out = intermediates.get("embedding_out")
    if embedding_out is not None and not torch.allclose(raw_states["resid_0"], embedding_out):
        raise RuntimeError("resid_0 does not match cache key 'embedding_out'.")

    final_block_key = f"blocks.{n_layer - 1}.output"
    if final_block_key not in intermediates:
        raise RuntimeError(
            f"Missing expected cache key {final_block_key!r}. "
            f"Available keys: {sorted(intermediates.keys())}"
        )
    final_block_output = intermediates[final_block_key]
    if final_block_output.shape != (batch_size, seq_len, n_embd):
        raise RuntimeError(
            f"Cache key {final_block_key!r} has shape {tuple(final_block_output.shape)}; "
            f"expected {(batch_size, seq_len, n_embd)}."
        )

    resid_final = model.ln_f(final_block_output)
    logits = outputs.get("logits")
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError("Model outputs are missing 'logits'.")
    recomputed_logits = model.lm_head(resid_final)
    _validate_resid_final_logits_match(recomputed_logits, logits)
    raw_states["resid_final"] = resid_final

    missing_states = set(expected_states) - set(raw_states.keys())
    if missing_states:
        raise RuntimeError(f"Failed to extract expected states: {sorted(missing_states)}")

    collections: StateCollections = {}
    if extract_mode == "target_only":
        assert target_positions is not None
        target_positions = target_positions.to(input_ids.device)
        collections[_TARGET_STATE_COLLECTION] = {
            name: _prepare_for_saving(_select_target_positions(tensor, target_positions), save_dtype)
            for name, tensor in raw_states.items()
        }
    elif extract_mode == "full_sequence":
        collections[_TARGET_STATE_COLLECTION] = {
            name: _prepare_for_saving(tensor, save_dtype)
            for name, tensor in raw_states.items()
        }
    elif extract_mode == "both":
        assert target_positions is not None
        target_positions = target_positions.to(input_ids.device)
        collections[_BOTH_TARGET_COLLECTION] = {
            name: _prepare_for_saving(_select_target_positions(tensor, target_positions), save_dtype)
            for name, tensor in raw_states.items()
        }
        collections[_BOTH_FULL_COLLECTION] = {
            name: _prepare_for_saving(tensor, save_dtype)
            for name, tensor in raw_states.items()
        }
    else:
        raise ValueError(f"Unsupported extract_mode={extract_mode!r}.")

    return collections


def accumulate_batch_outputs(
    accumulated: Dict[str, Dict[str, List[torch.Tensor]]],
    batch_outputs: StateCollections,
) -> None:
    """Append per-batch state tensors into ordered accumulation lists."""

    for collection_name, states in batch_outputs.items():
        collection_store = accumulated.setdefault(collection_name, {})
        for state_name, tensor in states.items():
            collection_store.setdefault(state_name, []).append(tensor)


def finalize_state_collections(
    accumulated: Dict[str, Dict[str, List[torch.Tensor]]],
) -> StateCollections:
    """Concatenate accumulated state tensors along the sample dimension."""

    finalized: StateCollections = {}
    for collection_name, states in accumulated.items():
        finalized[collection_name] = {
            state_name: torch.cat(chunks, dim=0)
            for state_name, chunks in states.items()
        }
    return finalized


def _build_state_definitions(n_layer: int) -> Dict[str, str]:
    definitions = {
        f"resid_{layer_idx}": (
            f"Input to block {layer_idx} (cache key 'blocks.{layer_idx}.input'). "
            "For layer 0 this equals embedding_out = token embedding + positional embedding + dropout."
        )
        for layer_idx in range(n_layer)
    }
    definitions["resid_final"] = (
        "Output of the last block passed through model.ln_f. "
        "This is the exact tensor consumed by lm_head."
    )
    return definitions


def validate_artifact(
    artifact: Mapping[str, Any],
    analysis_set: Mapping[str, Any],
    *,
    n_layer: int,
    n_embd: int,
    seq_len: int,
    extract_mode: str,
    save_dtype: torch.dtype,
) -> None:
    """Run strict shape, completeness, dtype, finiteness, and ordering checks."""

    num_samples = int(analysis_set["input_ids"].shape[0])
    expected_state_names = set(_expected_state_names(n_layer))

    if artifact["sample_ids"] != list(analysis_set["sample_ids"]):
        raise ValueError("Artifact sample_ids do not match analysis-set order exactly.")
    if artifact["group_labels"] != list(analysis_set["group_labels"]):
        raise ValueError("Artifact group_labels do not match analysis-set order exactly.")
    if not torch.equal(artifact["target_positions"], analysis_set["target_positions"]):
        raise ValueError("Artifact target_positions do not match analysis set.")
    if not torch.equal(artifact["input_ids"], analysis_set["input_ids"]):
        raise ValueError("Artifact input_ids do not match analysis set.")

    labels = analysis_set.get("labels")
    if labels is None:
        if artifact.get("labels") is not None:
            raise ValueError("Artifact labels is populated, but analysis set did not contain labels.")
    else:
        if artifact.get("labels") is None:
            raise ValueError("Artifact labels is missing, but analysis set contained labels.")
        if not torch.equal(artifact["labels"], labels):
            raise ValueError("Artifact labels do not match analysis set.")

    collection_names = _collection_names_for_mode(extract_mode)
    for collection_name in collection_names:
        if collection_name not in artifact:
            raise ValueError(f"Artifact is missing required state collection {collection_name!r}.")
        states = artifact[collection_name]
        if not isinstance(states, Mapping):
            raise TypeError(f"Artifact field {collection_name!r} must be a mapping.")

        missing = expected_state_names - set(states.keys())
        if missing:
            raise ValueError(
                f"State collection {collection_name!r} is missing required keys: {sorted(missing)}. "
                f"Got: {sorted(states.keys())}"
            )

        for state_name in expected_state_names:
            tensor = states[state_name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{collection_name}[{state_name!r}] must be a torch.Tensor.")
            if tensor.device.type != "cpu":
                raise ValueError(f"{collection_name}[{state_name!r}] must be on CPU before saving.")
            if tensor.dtype != save_dtype:
                raise ValueError(
                    f"{collection_name}[{state_name!r}] has dtype {tensor.dtype}; "
                    f"expected {save_dtype}."
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(
                    f"{collection_name}[{state_name!r}] contains NaN or Inf values."
                )
            if int(tensor.shape[0]) != num_samples:
                raise ValueError(
                    f"{collection_name}[{state_name!r}] dim-0 {int(tensor.shape[0])} "
                    f"!= num_samples {num_samples}."
                )

            expected_shape = _expected_state_shape(
                collection_name=collection_name,
                extract_mode=extract_mode,
                num_samples=num_samples,
                seq_len=seq_len,
                n_embd=n_embd,
            )
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"{collection_name}[{state_name!r}] shape {tuple(tensor.shape)} "
                    f"!= expected {expected_shape} for extract_mode={extract_mode!r}."
                )

    logger.info("Artifact validation passed.")


def finalize_and_validate_artifact(
    *,
    analysis_set: Mapping[str, Any],
    analysis_set_path: Path,
    experiment: ExperimentConfig,
    model_meta: Mapping[str, Any],
    extract_mode: str,
    dtype_name: str,
    save_dtype: torch.dtype,
    device: torch.device,
    finalized_states: StateCollections,
) -> dict[str, Any]:
    """Assemble the final artifact dict and validate it before saving."""

    input_ids = analysis_set["input_ids"]
    labels = analysis_set.get("labels")
    num_samples, seq_len = (int(input_ids.shape[0]), int(input_ids.shape[1]))
    n_layer = int(experiment.model.n_layer)
    n_embd = int(experiment.model.n_embd)

    state_definitions = _build_state_definitions(n_layer)
    artifact: dict[str, Any] = {
        "meta": {
            "checkpoint_path": model_meta["checkpoint_path"],
            "config_path": model_meta["config_path"],
            "config_source": model_meta["config_source"],
            "config_validation_status": model_meta["config_validation_status"],
            "analysis_set_path": str(analysis_set_path),
            "analysis_set_meta": copy.deepcopy(analysis_set.get("meta", {})),
            "model_type": model_meta["model_type"],
            "model_config": copy.deepcopy(model_meta["model_config"]),
            "n_layer": n_layer,
            "n_embd": n_embd,
            "block_size": int(experiment.model.block_size),
            "extract_mode": extract_mode,
            "dtype_saved": dtype_name,
            "device_used": str(device),
            "num_samples": num_samples,
            "seq_len": seq_len,
            "checkpoint_step": model_meta.get("step"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "script_version": _SCRIPT_VERSION,
            "state_definition": state_definitions,
            "target_position_semantics": (
                "target_positions[i] indexes position t in input_ids[i]. "
                "The extracted residual state at position t is the representation used to predict labels[i, t]."
            ),
            "resid_final_definition": "post_ln_f_pre_lm_head",
            "state_collection_layout": (
                {"states": "[N, d_model] target-position states"}
                if extract_mode == "target_only"
                else (
                    {"states": "[N, T, d_model] full-sequence states"}
                    if extract_mode == "full_sequence"
                    else {
                        "states_target": "[N, d_model] target-position states",
                        "states_full": "[N, T, d_model] full-sequence states",
                    }
                )
            ),
        },
        "sample_ids": list(analysis_set["sample_ids"]),
        "group_labels": list(analysis_set["group_labels"]),
        "target_positions": analysis_set["target_positions"].clone(),
        "input_ids": input_ids.clone(),
        "labels": None if labels is None else labels.clone(),
    }

    if extract_mode == "both":
        artifact[_BOTH_TARGET_COLLECTION] = finalized_states[_BOTH_TARGET_COLLECTION]
        artifact[_BOTH_FULL_COLLECTION] = finalized_states[_BOTH_FULL_COLLECTION]
    else:
        artifact[_TARGET_STATE_COLLECTION] = finalized_states[_TARGET_STATE_COLLECTION]

    validate_artifact(
        artifact,
        analysis_set,
        n_layer=n_layer,
        n_embd=n_embd,
        seq_len=seq_len,
        extract_mode=extract_mode,
        save_dtype=save_dtype,
    )
    return artifact


def save_artifact(artifact: Mapping[str, Any], output_path: Path, overwrite: bool) -> None:
    """Save the extracted artifact as a ``.pt`` file."""

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(artifact), output_path)
    logger.info("Saved artifact to %s", output_path)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract residual-stream / block-input states for downstream CKA analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config.json path. If omitted, the script tries to auto-detect config.json near the checkpoint.",
    )
    parser.add_argument(
        "--analysis-set",
        type=str,
        required=True,
        help="Path to an analysis-set .pt file generated by analysis/make_analysis_set.py.",
    )
    parser.add_argument("--output", type=str, required=True, help="Output artifact path (.pt).")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: cpu, cuda, cuda:0, or auto (default: auto).",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for forward passes (default: 32).")
    parser.add_argument(
        "--extract-mode",
        type=str,
        default="target_only",
        choices=["target_only", "full_sequence", "both"],
        help=(
            "target_only saves [N, d_model]; full_sequence saves [N, T, d_model]; "
            "both saves both collections."
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Dtype used for saved activation tensors (default: float32).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count for tensor-backed batching (default: 0).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite --output if it already exists.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")

    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer.")
    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0.")
    return args


def _log_saved_states(artifact: Mapping[str, Any], extract_mode: str) -> None:
    collection_names = _collection_names_for_mode(extract_mode)
    logger.info("Saved state collections for extract_mode=%s:", extract_mode)
    for collection_name in collection_names:
        logger.info("  %s:", collection_name)
        states = artifact[collection_name]
        for state_name in sorted(states.keys(), key=_state_sort_key):
            logger.info("    %s: %s", state_name, list(states[state_name].shape))


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(name)s] %(message)s",
    )

    device = resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    save_dtype = _resolve_dtype(args.dtype)

    analysis_set_path = Path(args.analysis_set).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    logger.info("checkpoint: %s", Path(args.checkpoint).expanduser().resolve())
    logger.info("analysis_set: %s", analysis_set_path)
    logger.info("output: %s", output_path)
    logger.info("device: %s", device)
    logger.info("batch_size: %d", args.batch_size)
    logger.info("extract_mode: %s", args.extract_mode)
    logger.info("dtype_saved: %s", args.dtype)

    analysis_set = load_analysis_set(analysis_set_path)
    input_ids = analysis_set["input_ids"]
    target_positions = analysis_set["target_positions"]
    labels = analysis_set.get("labels")
    num_samples, seq_len = (int(input_ids.shape[0]), int(input_ids.shape[1]))
    logger.info(
        "analysis samples: %d | seq_len: %d | target_positions range=[%d, %d]",
        num_samples,
        seq_len,
        int(target_positions.min().item()),
        int(target_positions.max().item()),
    )
    if labels is None:
        logger.info("analysis set does not contain labels; artifact will save labels=None.")
    else:
        logger.info("analysis set contains labels with shape %s.", list(labels.shape))

    model, experiment, model_meta = load_model_and_config(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=device,
    )
    logger.info("model_type: %s", model_meta["model_type"])
    logger.info("n_layer: %d", model_meta["n_layer"])
    logger.info("n_embd: %d", model_meta["n_embd"])
    logger.info("checkpoint_step: %s", model_meta.get("step"))
    logger.info("config_path: %s", model_meta["config_path"])
    logger.info("config_source: %s", model_meta["config_source"])
    logger.info("config_validation_status: %s", model_meta["config_validation_status"])

    if seq_len > int(experiment.model.block_size):
        raise ValueError(
            f"analysis set seq_len={seq_len} exceeds checkpoint block_size={experiment.model.block_size}."
        )
    if num_samples != len(analysis_set["sample_ids"]):
        raise ValueError(
            f"input_ids.shape[0]={num_samples} does not match len(sample_ids)={len(analysis_set['sample_ids'])}."
        )
    if len(analysis_set["group_labels"]) != num_samples:
        raise ValueError(
            f"len(group_labels)={len(analysis_set['group_labels'])} does not match num_samples={num_samples}."
        )

    dataset = AnalysisTensorDataset(input_ids=input_ids, target_positions=target_positions)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    accumulated: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    total_batches = len(dataloader)
    for batch_index, (batch_input_ids, batch_target_positions) in enumerate(dataloader, start=1):
        batch_input_ids = batch_input_ids.to(device, non_blocking=(device.type == "cuda"))
        batch_tp = batch_target_positions if args.extract_mode in {"target_only", "both"} else None
        batch_outputs = extract_states_for_batch(
            model=model,
            input_ids=batch_input_ids,
            extract_mode=args.extract_mode,
            target_positions=batch_tp,
            save_dtype=save_dtype,
        )
        accumulate_batch_outputs(accumulated, batch_outputs)
        logger.debug("batch %d/%d extracted successfully.", batch_index, total_batches)

    finalized_states = finalize_state_collections(accumulated)
    artifact = finalize_and_validate_artifact(
        analysis_set=analysis_set,
        analysis_set_path=analysis_set_path,
        experiment=experiment,
        model_meta=model_meta,
        extract_mode=args.extract_mode,
        dtype_name=args.dtype,
        save_dtype=save_dtype,
        device=device,
        finalized_states=finalized_states,
    )
    save_artifact(artifact, output_path=output_path, overwrite=args.overwrite)
    _log_saved_states(artifact, extract_mode=args.extract_mode)

    print(flush=True)
    print(f"[extract_residuals] checkpoint: {model_meta['checkpoint_path']}", flush=True)
    print(f"[extract_residuals] config: {model_meta['config_path']}", flush=True)
    print(f"[extract_residuals] model_type: {model_meta['model_type']}", flush=True)
    print(f"[extract_residuals] n_layer: {model_meta['n_layer']}", flush=True)
    print(f"[extract_residuals] n_embd: {model_meta['n_embd']}", flush=True)
    print(f"[extract_residuals] analysis samples: {num_samples}", flush=True)
    print(f"[extract_residuals] extract_mode: {args.extract_mode}", flush=True)
    print(f"[extract_residuals] dtype_saved: {args.dtype}", flush=True)
    print(f"[extract_residuals] output: {output_path}", flush=True)
    if args.extract_mode == "both":
        for collection_name in _collection_names_for_mode(args.extract_mode):
            print(f"[extract_residuals] saved {collection_name}:", flush=True)
            for state_name in sorted(artifact[collection_name].keys(), key=_state_sort_key):
                print(
                    f"  - {state_name}: {list(artifact[collection_name][state_name].shape)}",
                    flush=True,
                )
    else:
        print(f"[extract_residuals] saved {_TARGET_STATE_COLLECTION} ({args.extract_mode}):", flush=True)
        for state_name in sorted(artifact[_TARGET_STATE_COLLECTION].keys(), key=_state_sort_key):
            print(
                f"  - {state_name}: {list(artifact[_TARGET_STATE_COLLECTION][state_name].shape)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
