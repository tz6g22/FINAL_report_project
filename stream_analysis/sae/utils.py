"""Shared utility helpers for the SAE subproject."""

from __future__ import annotations

import csv
import math
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def infer_project_root() -> Path:
    """Return the repository root from the SAE package location."""

    return Path(__file__).resolve().parents[2]


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it."""

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a simple process-wide logging format."""

    if logging.getLogger().handlers:
        logging.getLogger().setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def resolve_device(device: str) -> torch.device:
    """Resolve ``auto`` into a concrete torch device."""

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(name: str) -> torch.dtype:
    """Map a user-facing dtype string to ``torch.dtype``."""

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise KeyError(f"Unsupported dtype {name!r}.")
    return mapping[name]


def seed_everything(seed: int) -> None:
    """Set Python / NumPy / Torch RNG state."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(payload: Any, path: str | Path) -> Path:
    """Write JSON with UTF-8 encoding."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return target


def read_json(path: str | Path) -> Any:
    """Read JSON from disk."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sanitize_component(value: object) -> str:
    """Make a path component filesystem-safe without hiding its meaning."""

    text = str(value).strip()
    if not text:
        return "unknown"
    return _SANITIZE_PATTERN.sub("_", text)


def format_checkpoint_step(value: object) -> str:
    """Normalize checkpoint step identifiers into stable directory names."""

    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    return sanitize_component(text) if text else "unknown"


def default_activation_dir(model_type: str, checkpoint_step: object, layer_idx: int, site: str) -> Path:
    """Return the default activation shard directory."""

    return (
        infer_project_root()
        / "outputs"
        / "activations"
        / "sae"
        / sanitize_component(model_type)
        / f"step_{format_checkpoint_step(checkpoint_step)}"
        / f"layer_{layer_idx}"
        / sanitize_component(site)
    )


def default_sae_checkpoint_dir(model_type: str, checkpoint_step: object, layer_idx: int, site: str) -> Path:
    """Return the default SAE checkpoint directory."""

    return (
        infer_project_root()
        / "outputs"
        / "sae_checkpoints"
        / sanitize_component(model_type)
        / f"step_{format_checkpoint_step(checkpoint_step)}"
        / f"layer_{layer_idx}"
        / sanitize_component(site)
    )


def default_sae_eval_dir(model_type: str, checkpoint_step: object, layer_idx: int, site: str) -> Path:
    """Return the default SAE evaluation directory."""

    return (
        infer_project_root()
        / "outputs"
        / "sae_eval"
        / sanitize_component(model_type)
        / f"step_{format_checkpoint_step(checkpoint_step)}"
        / f"layer_{layer_idx}"
        / sanitize_component(site)
    )


def default_sae_analysis_dir(model_type: str, checkpoint_step: object, layer_idx: int, site: str) -> Path:
    """Return the default SAE analysis directory."""

    return (
        infer_project_root()
        / "outputs"
        / "sae_analysis"
        / sanitize_component(model_type)
        / f"step_{format_checkpoint_step(checkpoint_step)}"
        / f"layer_{layer_idx}"
        / sanitize_component(site)
    )


def default_sae_intervention_dir(model_type: str, checkpoint_step: object, layer_idx: int, site: str) -> Path:
    """Return the default SAE intervention directory."""

    return (
        infer_project_root()
        / "outputs"
        / "sae_intervention"
        / sanitize_component(model_type)
        / f"step_{format_checkpoint_step(checkpoint_step)}"
        / f"layer_{layer_idx}"
        / sanitize_component(site)
    )


def append_csv_row(path: str | Path, fieldnames: Iterable[str], row: Mapping[str, object]) -> Path:
    """Append one CSV row, writing a header when the file is new."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_header = not target.exists()
    with target.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        if write_header:
            writer.writeheader()
        writer.writerow(dict(row))
    return target


def write_csv_rows(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> Path:
    """Write a complete CSV table."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return target


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of row dicts."""

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def maybe_make_dataframe(rows: Sequence[Mapping[str, object]]):
    """Return a pandas DataFrame when pandas is available, else ``None``."""

    try:
        import pandas as pd
    except Exception:  # pragma: no cover - pandas is optional
        return None
    return pd.DataFrame(list(rows))


def sigmoid_safe_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute exact binary AUROC from raw scores without sklearn.

    ``labels`` must be a boolean / {0,1} vector where 1 indicates the positive
    class. This implementation handles ties using average ranks.
    """

    values = np.asarray(scores, dtype=float)
    targets = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or targets.ndim != 1 or values.shape[0] != targets.shape[0]:
        raise ValueError("scores and labels must be aligned 1D arrays.")

    pos_count = int((targets == 1).sum())
    neg_count = int((targets == 0).sum())
    if pos_count == 0 or neg_count == 0:
        return float("nan")

    order = np.argsort(values, kind="mergesort")
    sorted_scores = values[order]
    sorted_labels = targets[order]
    ranks = np.empty_like(sorted_scores, dtype=float)

    start = 0
    n = sorted_scores.shape[0]
    while start < n:
        end = start + 1
        while end < n and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[start:end] = avg_rank
        start = end

    pos_rank_sum = float(ranks[sorted_labels == 1].sum())
    auc = (pos_rank_sum - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count)
    return float(auc)


def finite_stats(values: Sequence[float]) -> dict[str, float]:
    """Return basic summary statistics over finite values."""

    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0.0,
            "mean": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": float(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def safe_float(value: object) -> float:
    """Convert a numeric-like value to float while preserving NaNs."""

    if isinstance(value, float):
        return value
    if isinstance(value, (int, np.integer)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            pass
    return float(value)


def format_top_pairs(
    pairs: Sequence[tuple[int, int, float]],
    *,
    limit: int,
) -> list[dict[str, float]]:
    """Convert latent-pair tuples into JSON-friendly records."""

    records = []
    for left, right, score in list(pairs)[:limit]:
        records.append(
            {
                "latent_i": float(left),
                "latent_j": float(right),
                "score": float(score),
            }
        )
    return records


def prepare_output_dir(
    output_dir: str | Path,
    *,
    overwrite: bool,
    allowed_existing_names: Iterable[str] | None = None,
) -> Path:
    """Prepare an output directory while avoiding accidental clobbering."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    existing = list(target.iterdir())
    if not existing:
        return target
    if not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {target}. Use overwrite=True to replace known artifacts."
        )

    allowed_prefixes = tuple(allowed_existing_names or ())
    for child in existing:
        if child.name == "meta.json" or child.name.startswith("shard_") or child.name in allowed_prefixes:
            if child.is_file():
                child.unlink()
            else:
                raise RuntimeError(
                    f"Refusing to overwrite non-file artifact inside output directory: {child}"
                )
            continue
        raise RuntimeError(
            f"Refusing to overwrite unexpected existing artifact in output directory: {child}"
        )
    return target
