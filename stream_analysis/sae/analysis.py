"""SAE analysis helpers for mem/nonmem feature studies."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch

from scripts.evaluate import load_checkpoint
from stream_analysis.extract_residuals import load_analysis_set

from .data import preprocess_activations
from .eval import (
    SAEEvaluator,
    compute_coactivation_from_latents,
    compute_decoder_overlap,
    load_sae_checkpoint,
    summarize_sae_eval,
)
from .extract import BlockInputExtractor
from .utils import (
    default_sae_analysis_dir,
    ensure_dir,
    format_checkpoint_step,
    read_csv_rows,
    read_json,
    resolve_device,
    safe_float,
    sanitize_component,
    sigmoid_safe_auc,
    write_csv_rows,
    write_json,
)

logger = logging.getLogger("sae.analysis")

_MEM_LABEL_ALIASES = {
    "mem",
    "memorized",
    "memorised",
    "member",
    "members",
    "train",
    "training",
    "generaltrain",
    "general_train",
    "intraining",
    "seen",
}
_NONMEM_LABEL_ALIASES = {
    "nonmem",
    "nonmemorized",
    "nonmemorised",
    "nonmember",
    "non_member",
    "heldout",
    "holdout",
    "unseen",
    "val",
    "validation",
    "generalval",
    "general_val",
    "test",
}


@dataclass
class LatentActivitySummary:
    """Compact per-latent usage summary."""

    latent_index: int
    firing_rate: float
    mean_activation: float
    max_activation: float


@dataclass
class MemLabelArtifact:
    """Normalized mem/nonmem labels plus aligned analysis-set tensors."""

    source_path: str
    sample_ids: List[str]
    group_labels: List[str]
    mem_labels: torch.Tensor
    input_ids: torch.Tensor
    target_positions: torch.Tensor
    labels: torch.Tensor | None = None
    texts: List[str] | None = None
    tokenizer_name: str | None = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_samples(self) -> int:
        return len(self.sample_ids)

    @property
    def mem_count(self) -> int:
        return int(self.mem_labels.sum().item())

    @property
    def nonmem_count(self) -> int:
        return int(self.num_samples - self.mem_count)


def summarize_latent_activity(latents: torch.Tensor, threshold: float = 0.0) -> List[LatentActivitySummary]:
    """Return per-latent activity summaries for a ``[batch, n_latents]`` tensor."""

    if latents.ndim != 2:
        raise ValueError(f"latents must have shape [batch, n_latents], got {tuple(latents.shape)}.")
    active = latents.abs() > threshold
    summaries: List[LatentActivitySummary] = []
    for latent_idx in range(latents.shape[1]):
        column = latents[:, latent_idx]
        summaries.append(
            LatentActivitySummary(
                latent_index=latent_idx,
                firing_rate=float(active[:, latent_idx].float().mean().item()),
                mean_activation=float(column.mean().item()),
                max_activation=float(column.max().item()),
            )
        )
    return summaries


def summarize_activity_as_dicts(latents: torch.Tensor, threshold: float = 0.0) -> List[Dict[str, float]]:
    """Convenience wrapper returning plain dicts for JSON / CSV export."""

    return [
        {
            "latent_index": summary.latent_index,
            "firing_rate": summary.firing_rate,
            "mean_activation": summary.mean_activation,
            "max_activation": summary.max_activation,
        }
        for summary in summarize_latent_activity(latents, threshold=threshold)
    ]


def _normalize_group_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", label.strip().lower())


def _infer_mem_label(label: str) -> int | None:
    normalized = _normalize_group_label(label)
    if normalized in _MEM_LABEL_ALIASES:
        return 1
    if normalized in _NONMEM_LABEL_ALIASES:
        return 0
    if "train" in normalized or "member" in normalized or "memor" in normalized:
        return 1
    if "val" in normalized or "held" in normalized or "nonmem" in normalized or "nonmember" in normalized:
        return 0
    return None


def _extract_tokenizer_name(payload: Mapping[str, Any]) -> str | None:
    meta = payload.get("meta")
    if isinstance(meta, Mapping):
        value = meta.get("tokenizer_name")
        if isinstance(value, str) and value.strip():
            return value
        analysis_meta = meta.get("analysis_set_meta")
        if isinstance(analysis_meta, Mapping):
            nested = analysis_meta.get("tokenizer_name")
            if isinstance(nested, str) and nested.strip():
                return nested
    return None


def _filtered_texts(values: object, indices: Sequence[int]) -> List[str] | None:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return None
    if len(values) < len(indices):
        return None
    output: List[str] = []
    for index in indices:
        value = values[index]
        output.append(value if isinstance(value, str) else str(value))
    return output


def load_mem_labels(labels_source: str | Path) -> MemLabelArtifact:
    """Load mem/nonmem labels from an existing analysis-set style artifact."""

    resolved = Path(labels_source).expanduser().resolve()
    payload = load_analysis_set(resolved)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Labels source must deserialize to a mapping, got {type(payload).__name__}.")

    sample_ids = list(payload["sample_ids"])
    group_labels = list(payload["group_labels"])
    input_ids = payload["input_ids"].detach().cpu().to(torch.long)
    target_positions = payload["target_positions"].detach().cpu().to(torch.long)
    labels = payload.get("labels")
    if labels is not None:
        if not torch.is_tensor(labels):
            raise TypeError("labels_source['labels'] must be a torch.Tensor when present.")
        labels = labels.detach().cpu().to(torch.long)

    mem_values: List[int] = []
    keep_indices: List[int] = []
    skipped_labels: Dict[str, int] = {}
    for index, label in enumerate(group_labels):
        inferred = _infer_mem_label(label)
        if inferred is None:
            skipped_labels[label] = skipped_labels.get(label, 0) + 1
            continue
        keep_indices.append(index)
        mem_values.append(inferred)

    if skipped_labels:
        logger.warning(
            "skipping %d analysis samples with unresolved labels: %s",
            sum(skipped_labels.values()),
            dict(sorted(skipped_labels.items())),
        )

    if not keep_indices:
        raise ValueError(
            "No mem/nonmem labels could be resolved from group_labels. "
            "Expected labels such as train/val, general_train/general_val, mem/nonmem."
        )

    keep_tensor = torch.tensor(keep_indices, dtype=torch.long)
    filtered_sample_ids = [sample_ids[index] for index in keep_indices]
    filtered_group_labels = [group_labels[index] for index in keep_indices]
    filtered_input_ids = input_ids.index_select(0, keep_tensor)
    filtered_target_positions = target_positions.index_select(0, keep_tensor)
    filtered_labels = None if labels is None else labels.index_select(0, keep_tensor)

    mem_tensor = torch.tensor(mem_values, dtype=torch.long)
    mem_count = int(mem_tensor.sum().item())
    nonmem_count = int(mem_tensor.shape[0] - mem_count)
    if mem_count == 0 or nonmem_count == 0:
        raise ValueError(
            "Mem analysis requires both mem and nonmem examples after label normalization. "
            f"Observed mem={mem_count}, nonmem={nonmem_count}."
        )

    texts = _filtered_texts(payload.get("texts"), keep_indices)
    return MemLabelArtifact(
        source_path=str(resolved),
        sample_ids=filtered_sample_ids,
        group_labels=filtered_group_labels,
        mem_labels=mem_tensor,
        input_ids=filtered_input_ids,
        target_positions=filtered_target_positions,
        labels=filtered_labels,
        texts=texts,
        tokenizer_name=_extract_tokenizer_name(payload),
        meta=dict(payload.get("meta", {})) if isinstance(payload.get("meta"), Mapping) else {},
    )


def compute_feature_selectivity(
    latents: torch.Tensor,
    mem_labels: torch.Tensor,
    *,
    threshold: float = 1e-8,
) -> List[Dict[str, float]]:
    """Compute per-feature mem vs nonmem activation summaries."""

    if latents.ndim != 2:
        raise ValueError(f"latents must have shape [num_samples, n_latents], got {tuple(latents.shape)}.")
    if mem_labels.ndim != 1 or int(mem_labels.shape[0]) != int(latents.shape[0]):
        raise ValueError("mem_labels must be a 1D tensor aligned with latents.")

    mem_mask = mem_labels.to(dtype=torch.bool)
    nonmem_mask = ~mem_mask
    if int(mem_mask.sum().item()) == 0 or int(nonmem_mask.sum().item()) == 0:
        raise ValueError("Both mem and nonmem samples are required for selectivity stats.")

    latents = latents.to(dtype=torch.float32)
    mem_latents = latents[mem_mask]
    nonmem_latents = latents[nonmem_mask]

    mem_mean = mem_latents.mean(dim=0)
    nonmem_mean = nonmem_latents.mean(dim=0)
    diff = mem_mean - nonmem_mean
    mem_var = mem_latents.var(dim=0, unbiased=False)
    nonmem_var = nonmem_latents.var(dim=0, unbiased=False)
    pooled = ((mem_var + nonmem_var) * 0.5).sqrt().clamp_min(1e-8)
    effect_size = diff / pooled

    overall_mean = latents.mean(dim=0)
    overall_abs_mean = latents.abs().mean(dim=0)
    mem_active = (mem_latents.abs() > threshold).float().mean(dim=0)
    nonmem_active = (nonmem_latents.abs() > threshold).float().mean(dim=0)
    overall_active = (latents.abs() > threshold).float().mean(dim=0)

    rows: List[Dict[str, float]] = []
    for latent_index in range(latents.shape[1]):
        rows.append(
            {
                "latent_index": float(latent_index),
                "mean_mem": float(mem_mean[latent_index].item()),
                "mean_nonmem": float(nonmem_mean[latent_index].item()),
                "diff": float(diff[latent_index].item()),
                "standardized_effect_size": float(effect_size[latent_index].item()),
                "mem_active_rate": float(mem_active[latent_index].item()),
                "nonmem_active_rate": float(nonmem_active[latent_index].item()),
                "overall_active_rate": float(overall_active[latent_index].item()),
                "overall_mean": float(overall_mean[latent_index].item()),
                "overall_abs_mean": float(overall_abs_mean[latent_index].item()),
            }
        )
    return rows


def compute_feature_mem_auc(
    latents: torch.Tensor,
    mem_labels: torch.Tensor,
) -> List[Dict[str, float]]:
    """Compute a per-feature AUROC for mem vs nonmem prediction."""

    if latents.ndim != 2:
        raise ValueError(f"latents must have shape [num_samples, n_latents], got {tuple(latents.shape)}.")
    if mem_labels.ndim != 1 or int(mem_labels.shape[0]) != int(latents.shape[0]):
        raise ValueError("mem_labels must be a 1D tensor aligned with latents.")

    scores = latents.detach().cpu().to(dtype=torch.float32).numpy()
    labels = mem_labels.detach().cpu().to(dtype=torch.int64).numpy()
    rows: List[Dict[str, float]] = []
    for latent_index in range(scores.shape[1]):
        rows.append(
            {
                "latent_index": float(latent_index),
                "mem_auc": float(sigmoid_safe_auc(scores[:, latent_index], labels)),
            }
        )
    return rows


def _load_optional_tokenizer(tokenizer_name: str | None):
    if tokenizer_name is None:
        return None
    try:
        from transformers import AutoTokenizer
    except Exception as error:  # pragma: no cover - optional dependency
        logger.warning("transformers is unavailable; token decoding will be skipped: %s", error)
        return None
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, local_files_only=True)
    except Exception as error:  # pragma: no cover - depends on local cache
        logger.warning("could not load tokenizer %s locally; token decoding will be skipped: %s", tokenizer_name, error)
        return None


def top_activating_examples(
    latents: torch.Tensor,
    label_artifact: MemLabelArtifact,
    *,
    topk: int = 10,
    threshold: float = 1e-8,
) -> Dict[str, List[Dict[str, Any]]]:
    """Collect top-activating examples for every feature."""

    if latents.ndim != 2:
        raise ValueError(f"latents must have shape [num_samples, n_latents], got {tuple(latents.shape)}.")
    if int(latents.shape[0]) != label_artifact.num_samples:
        raise ValueError("latents and label_artifact must describe the same number of samples.")

    tokenizer = _load_optional_tokenizer(label_artifact.tokenizer_name)
    latents_cpu = latents.detach().cpu().to(dtype=torch.float32)
    limit = min(topk, int(latents_cpu.shape[0]))
    if limit <= 0:
        return {}

    output: Dict[str, List[Dict[str, Any]]] = {}
    for latent_index in range(latents_cpu.shape[1]):
        column = latents_cpu[:, latent_index]
        active_count = int((column.abs() > threshold).sum().item())
        if active_count == 0:
            output[str(latent_index)] = []
            continue

        values, indices = torch.topk(column, k=limit)
        records: List[Dict[str, Any]] = []
        for rank, (value, sample_index) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
            token_idx = int(label_artifact.target_positions[sample_index].item())
            raw_token_id = int(label_artifact.input_ids[sample_index, token_idx].item())
            record: Dict[str, Any] = {
                "rank": rank,
                "sample_id": label_artifact.sample_ids[sample_index],
                "group_label": label_artifact.group_labels[sample_index],
                "is_mem": bool(label_artifact.mem_labels[sample_index].item()),
                "token_idx": token_idx,
                "raw_token_id": raw_token_id,
                "activation": float(value),
            }
            if label_artifact.labels is not None:
                record["target_token_id"] = int(label_artifact.labels[sample_index, token_idx].item())
            if label_artifact.texts is not None and sample_index < len(label_artifact.texts):
                record["context_text"] = label_artifact.texts[sample_index]
            if tokenizer is not None:
                try:
                    record["token_text"] = tokenizer.decode([raw_token_id], skip_special_tokens=False)
                except Exception:
                    pass
                if "target_token_id" in record:
                    try:
                        record["target_token_text"] = tokenizer.decode(
                            [int(record["target_token_id"])],
                            skip_special_tokens=False,
                        )
                    except Exception:
                        pass
            records.append(record)
        output[str(latent_index)] = records
    return output


def summarize_mem_features(
    feature_stats_rows: Sequence[Mapping[str, object]],
    feature_auc_rows: Sequence[Mapping[str, object]],
    *,
    auc_margin: float = 0.10,
    topk: int = 10,
) -> Dict[str, object]:
    """Summarize the most mem-selective and nonmem-selective features."""

    auc_by_index = {
        int(float(row["latent_index"])): safe_float(row["mem_auc"])
        for row in feature_auc_rows
        if "latent_index" in row and "mem_auc" in row
    }
    merged: List[Dict[str, float]] = []
    for row in feature_stats_rows:
        latent_index = int(float(row["latent_index"]))
        merged_row = {key: safe_float(value) for key, value in row.items()}
        merged_row["latent_index"] = float(latent_index)
        merged_row["mem_auc"] = auc_by_index.get(latent_index, float("nan"))
        merged_row["auc_centered"] = merged_row["mem_auc"] - 0.5
        merged_row["shared_score"] = merged_row["overall_mean"] / (1.0 + abs(merged_row["standardized_effect_size"]))
        merged.append(merged_row)

    mem_selective = [
        row for row in merged if row["diff"] > 0.0 and row["mem_auc"] >= 0.5 + auc_margin
    ]
    nonmem_selective = [
        row for row in merged if row["diff"] < 0.0 and row["mem_auc"] <= 0.5 - auc_margin
    ]

    def _top_rows(rows: Iterable[Dict[str, float]], *, key_name: str, reverse: bool) -> List[Dict[str, float]]:
        ordered = sorted(rows, key=lambda row: row[key_name], reverse=reverse)
        return ordered[:topk]

    return {
        "mem_selective_feature_count": int(len(mem_selective)),
        "nonmem_selective_feature_count": int(len(nonmem_selective)),
        "most_mem_selective_features": _top_rows(mem_selective or merged, key_name="standardized_effect_size", reverse=True),
        "most_nonmem_selective_features": _top_rows(nonmem_selective or merged, key_name="standardized_effect_size", reverse=False),
        "shared_high_activity_features": _top_rows(merged, key_name="shared_score", reverse=True),
    }


def _resolve_source_metadata(
    sae_payload: Mapping[str, Any],
    activation_dir: str | Path,
) -> Dict[str, Any]:
    activation_meta = read_json(Path(activation_dir).expanduser().resolve() / "meta.json")
    if not isinstance(activation_meta, Mapping):
        raise TypeError("activation meta.json must contain a JSON object.")

    source_meta = sae_payload.get("activation_metadata")
    merged = dict(activation_meta)
    if isinstance(source_meta, Mapping):
        merged.update(dict(source_meta))
    checkpoint_path = merged.get("checkpoint_path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
        raise KeyError("Could not resolve source model checkpoint path from SAE checkpoint or activation meta.")
    return merged


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(subvalue) for key, subvalue in value.items() if key != "dataframe"}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value.__class__.__name__ == "DataFrame" and hasattr(value, "to_dict"):
        try:
            return value.to_dict(orient="records")
        except Exception:
            return None
    return value


def _select_target_rows(site_tensor: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
    batch_size = int(site_tensor.shape[0])
    return site_tensor[torch.arange(batch_size, dtype=torch.long), target_positions]


def _extract_target_activations(
    source_model: torch.nn.Module,
    label_artifact: MemLabelArtifact,
    *,
    layer_idx: int,
    site: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    extractor = BlockInputExtractor(source_model, layer_idx=layer_idx, site=site, device=device)
    batches: List[torch.Tensor] = []
    num_samples = label_artifact.num_samples
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_inputs = label_artifact.input_ids[start:end]
        batch_tp = label_artifact.target_positions[start:end]
        site_tensor = extractor.extract_site_tensor(batch_inputs)
        batches.append(_select_target_rows(site_tensor, batch_tp))
    if not batches:
        raise RuntimeError("No analysis activations were extracted.")
    return torch.cat(batches, dim=0)


def _reconstruction_summary_from_outputs(
    inputs: torch.Tensor,
    latents: torch.Tensor,
    recon_error: torch.Tensor,
    *,
    dead_threshold: float,
) -> Dict[str, float]:
    if inputs.ndim != 2 or latents.ndim != 2 or recon_error.ndim != 2:
        raise ValueError("inputs, latents, and recon_error must all be 2D tensors.")

    total_elements = int(inputs.numel())
    if total_elements <= 0:
        raise RuntimeError("Cannot summarize an empty reconstruction batch.")

    recon_mse = float(recon_error.pow(2).mean().item())
    mean_input = float(inputs.mean().item())
    mean_input_sq = float(inputs.pow(2).mean().item())
    input_variance = max(0.0, mean_input_sq - mean_input**2)
    active = latents.abs() > dead_threshold
    fired = active.any(dim=0)
    dead = ~fired
    return {
        "num_samples": float(inputs.shape[0]),
        "num_elements": float(total_elements),
        "recon_mse": recon_mse,
        "normalized_recon_mse": float(recon_mse / input_variance) if input_variance > 0.0 else float("nan"),
        "input_variance": float(input_variance),
        "avg_l0": float(active.float().sum(dim=1).mean().item()),
        "dead_latent_count": float(dead.sum().item()),
        "alive_latent_count": float(fired.sum().item()),
        "dead_latent_frac": float(dead.float().mean().item()),
    }


def save_feature_payload(
    out_dir: str | Path,
    *,
    latents: torch.Tensor,
    recon_error: torch.Tensor,
    feature_mask: torch.Tensor,
    label_artifact: MemLabelArtifact,
    metadata: Mapping[str, object],
    filename: str = "feature_payload.pt",
) -> Path:
    """Persist per-sample SAE outputs for future intervention work."""

    output_dir = ensure_dir(out_dir)
    payload = {
        "z": latents.detach().cpu().to(torch.float32),
        "recon_error": recon_error.detach().cpu().to(torch.float32),
        "feature_mask": feature_mask.detach().cpu().to(torch.bool),
        "sample_ids": list(label_artifact.sample_ids),
        "group_labels": list(label_artifact.group_labels),
        "mem_labels": label_artifact.mem_labels.detach().cpu().to(torch.long),
        "target_positions": label_artifact.target_positions.detach().cpu().to(torch.long),
        "input_ids": label_artifact.input_ids.detach().cpu().to(torch.long),
        "labels": None if label_artifact.labels is None else label_artifact.labels.detach().cpu().to(torch.long),
        "meta": dict(metadata),
    }
    path = output_dir / filename
    torch.save(payload, path)
    return path


def discover_analysis_dirs(path: str | Path) -> List[Path]:
    """Discover SAE analysis directories under a root or accept one run directory directly."""

    resolved = Path(path).expanduser().resolve()
    if resolved.is_file():
        if resolved.name != "eval_summary.json":
            raise ValueError(f"Expected eval_summary.json when passing a file path, got {resolved.name}.")
        return [resolved.parent]
    if (resolved / "eval_summary.json").is_file():
        return [resolved]
    return sorted(item.parent for item in resolved.rglob("eval_summary.json"))


def load_analysis_summary(analysis_dir: str | Path) -> Dict[str, object]:
    """Load one saved analysis summary and flatten key comparison metrics."""

    directory = Path(analysis_dir).expanduser().resolve()
    payload = read_json(directory / "eval_summary.json")
    if not isinstance(payload, Mapping):
        raise TypeError(f"eval_summary.json must contain a JSON object, got {type(payload).__name__}.")

    summary = payload.get("summary", {})
    feature_summary = payload.get("feature_summary", {})
    if not isinstance(summary, Mapping):
        raise TypeError("eval_summary.json['summary'] must be a mapping.")
    if not isinstance(feature_summary, Mapping):
        raise TypeError("eval_summary.json['feature_summary'] must be a mapping.")

    return {
        "analysis_dir": str(directory),
        "model_type": summary.get("model_type", payload.get("model_type", "unknown")),
        "checkpoint_step": summary.get("checkpoint_step", payload.get("checkpoint_step")),
        "layer_idx": summary.get("layer_idx", payload.get("layer_idx")),
        "site": summary.get("site", payload.get("site", "input")),
        "k": summary.get("k"),
        "n_latents": summary.get("n_latents"),
        "normalized_recon_mse": summary.get("normalized_recon_mse"),
        "avg_l0": summary.get("avg_l0"),
        "dead_latent_frac": summary.get("dead_latent_frac"),
        "mem_selective_feature_count": feature_summary.get("mem_selective_feature_count"),
        "nonmem_selective_feature_count": feature_summary.get("nonmem_selective_feature_count"),
        "decoder_overlap_mean": summary.get("decoder_overlap_mean"),
        "decoder_overlap_median": summary.get("decoder_overlap_median"),
        "decoder_overlap_max": summary.get("decoder_overlap_max"),
        "coactivation_mean": summary.get("coactivation_mean"),
        "coactivation_median": summary.get("coactivation_median"),
        "coactivation_max": summary.get("coactivation_max"),
    }


def build_comparison_rows(paths: Sequence[str | Path]) -> List[Dict[str, object]]:
    """Build a flat comparison table from one or more analysis directories / roots."""

    directories: List[Path] = []
    for path in paths:
        directories.extend(discover_analysis_dirs(path))

    seen: set[Path] = set()
    unique_directories: List[Path] = []
    for directory in directories:
        if directory in seen:
            continue
        seen.add(directory)
        unique_directories.append(directory)

    rows = [load_analysis_summary(directory) for directory in unique_directories]
    rows.sort(
        key=lambda row: (
            sanitize_component(row.get("model_type", "unknown")),
            format_checkpoint_step(row.get("checkpoint_step")),
            int(row.get("layer_idx", 0)),
            sanitize_component(row.get("site", "input")),
        )
    )
    return rows


def save_comparison_rows(
    rows: Sequence[Mapping[str, object]],
    out_dir: str | Path,
    *,
    filename: str = "comparison_metrics.csv",
) -> Path:
    """Persist a standardized comparison table to CSV."""

    fieldnames = [
        "analysis_dir",
        "model_type",
        "checkpoint_step",
        "layer_idx",
        "site",
        "k",
        "n_latents",
        "normalized_recon_mse",
        "avg_l0",
        "dead_latent_frac",
        "mem_selective_feature_count",
        "nonmem_selective_feature_count",
        "decoder_overlap_mean",
        "decoder_overlap_median",
        "decoder_overlap_max",
        "coactivation_mean",
        "coactivation_median",
        "coactivation_max",
    ]
    output_dir = ensure_dir(out_dir)
    return write_csv_rows(output_dir / filename, fieldnames, rows)


def load_feature_stats_rows(analysis_dir: str | Path) -> List[Dict[str, float]]:
    """Load ``feature_stats.csv`` from an SAE analysis directory."""

    directory = Path(analysis_dir).expanduser().resolve()
    rows = read_csv_rows(directory / "feature_stats.csv")
    output: List[Dict[str, float]] = []
    for row in rows:
        output.append({key: safe_float(value) for key, value in row.items()})
    return output


def select_top_feature_ids(
    feature_stats_rows: Sequence[Mapping[str, object]],
    *,
    selection: str,
    topn: int,
) -> List[int]:
    """Select top feature ids from saved mem/nonmem feature statistics."""

    if topn <= 0:
        raise ValueError("topn must be positive.")
    normalized_selection = str(selection).strip().lower()
    if normalized_selection not in {"top_mem", "top_nonmem"}:
        raise ValueError("selection must be 'top_mem' or 'top_nonmem'.")

    rows = list(feature_stats_rows)
    if normalized_selection == "top_mem":
        ordered = sorted(rows, key=lambda row: safe_float(row["standardized_effect_size"]), reverse=True)
    else:
        ordered = sorted(rows, key=lambda row: safe_float(row["standardized_effect_size"]))
    return [int(safe_float(row["latent_index"])) for row in ordered[:topn]]


def run_sae_mem_analysis(
    sae_checkpoint: str | Path,
    activation_dir: str | Path,
    labels_source: str | Path,
    *,
    batch_size: int = 128,
    device: str | torch.device = "auto",
    out_dir: str | Path | None = None,
    topk_examples: int = 10,
    dead_threshold: float = 1e-8,
) -> Dict[str, object]:
    """Run held-out SAE evaluation plus mem/nonmem feature analysis."""

    torch_device = resolve_device(str(device)) if not isinstance(device, torch.device) else device
    sae_model, sae_config, sae_payload = load_sae_checkpoint(sae_checkpoint, device=torch_device)
    label_artifact = load_mem_labels(labels_source)
    source_meta = _resolve_source_metadata(sae_payload, activation_dir)

    model_type = str(source_meta.get("model_type", "unknown"))
    checkpoint_step = source_meta.get("checkpoint_step")
    layer_idx = int(source_meta.get("layer_idx", 0))
    site = str(source_meta.get("site", "input"))
    checkpoint_path = Path(str(source_meta["checkpoint_path"])).expanduser().resolve()

    train_config = sae_payload.get("train_config", {})
    preprocessing = "none"
    if isinstance(train_config, Mapping):
        preprocessing = str(train_config.get("preprocessing", "none"))

    source_model, _, _ = load_checkpoint(checkpoint_path, device=torch_device)
    raw_inputs = _extract_target_activations(
        source_model,
        label_artifact,
        layer_idx=layer_idx,
        site=site,
        batch_size=batch_size,
        device=torch_device,
    )
    processed_inputs = preprocess_activations(raw_inputs, mode=preprocessing)

    sae_inputs = processed_inputs.to(device=torch_device, dtype=sae_model.b_pre.dtype)
    with torch.no_grad():
        outputs = sae_model(sae_inputs)
    latents = outputs["z"].detach().cpu().to(torch.float32)
    recon_error = outputs["recon_error"].detach().cpu().to(torch.float32)
    feature_mask = latents.abs() > dead_threshold

    reconstruction = _reconstruction_summary_from_outputs(
        processed_inputs.cpu(),
        latents,
        recon_error,
        dead_threshold=dead_threshold,
    )
    decoder_overlap = compute_decoder_overlap(sae_model)
    coactivation = compute_coactivation_from_latents(latents, threshold=dead_threshold)

    evaluator = SAEEvaluator(sae_model, sae_config, device=torch_device)
    activation_dir_summary = evaluator.summarize_activation_dir(
        activation_dir,
        batch_size=batch_size,
        preprocessing=preprocessing,
        num_workers=0,
        max_batches=None,
        dead_threshold=dead_threshold,
    )

    feature_stats_rows = compute_feature_selectivity(latents, label_artifact.mem_labels, threshold=dead_threshold)
    feature_auc_rows = compute_feature_mem_auc(latents, label_artifact.mem_labels)
    feature_summary = summarize_mem_features(feature_stats_rows, feature_auc_rows)
    top_examples = top_activating_examples(
        latents,
        label_artifact,
        topk=topk_examples,
        threshold=dead_threshold,
    )

    combined = summarize_sae_eval(
        reconstruction,
        decoder_overlap,
        coactivation,
        context={
            "model_type": model_type,
            "checkpoint_step": checkpoint_step,
            "layer_idx": layer_idx,
            "site": site,
            "activation_dir": str(Path(activation_dir).expanduser().resolve()),
            "labels_source": str(Path(labels_source).expanduser().resolve()),
            "sae_checkpoint": str(Path(sae_checkpoint).expanduser().resolve()),
            "preprocessing": preprocessing,
            "d_in": int(processed_inputs.shape[1]),
            "n_latents": sae_config.n_latents,
            "k": sae_config.k,
            "num_analysis_samples": label_artifact.num_samples,
            "num_mem_samples": label_artifact.mem_count,
            "num_nonmem_samples": label_artifact.nonmem_count,
        },
    )
    summary = dict(combined["summary"])
    summary["mem_selective_feature_count"] = feature_summary["mem_selective_feature_count"]
    summary["nonmem_selective_feature_count"] = feature_summary["nonmem_selective_feature_count"]

    metrics_rows = list(combined["metrics_rows"])
    metrics_rows.extend(
        [
            {"metric": "mem_selective_feature_count", "value": float(feature_summary["mem_selective_feature_count"])},
            {"metric": "nonmem_selective_feature_count", "value": float(feature_summary["nonmem_selective_feature_count"])},
        ]
    )

    target_out_dir = (
        Path(out_dir).expanduser().resolve()
        if out_dir is not None
        else default_sae_analysis_dir(model_type, checkpoint_step, layer_idx, site)
    )
    ensure_dir(target_out_dir)

    payload = _json_safe(
        {
        "summary": summary,
        "metrics_rows": metrics_rows,
        "reconstruction": reconstruction,
        "decoder_overlap": decoder_overlap,
        "coactivation": coactivation,
        "feature_summary": feature_summary,
        "activation_dir_summary": activation_dir_summary,
        }
    )

    eval_summary_path = write_json(payload, target_out_dir / "eval_summary.json")
    eval_metrics_path = write_csv_rows(target_out_dir / "eval_metrics.csv", ["metric", "value"], metrics_rows)
    feature_stats_path = write_csv_rows(
        target_out_dir / "feature_stats.csv",
        [
            "latent_index",
            "mean_mem",
            "mean_nonmem",
            "diff",
            "standardized_effect_size",
            "mem_active_rate",
            "nonmem_active_rate",
            "overall_active_rate",
            "overall_mean",
            "overall_abs_mean",
        ],
        feature_stats_rows,
    )
    feature_auc_path = write_csv_rows(
        target_out_dir / "feature_auc.csv",
        ["latent_index", "mem_auc"],
        feature_auc_rows,
    )
    top_examples_path = write_json(top_examples, target_out_dir / "top_examples.json")
    feature_payload_path = save_feature_payload(
        target_out_dir,
        latents=latents,
        recon_error=recon_error,
        feature_mask=feature_mask,
        label_artifact=label_artifact,
        metadata={
            "model_type": model_type,
            "checkpoint_step": checkpoint_step,
            "layer_idx": layer_idx,
            "site": site,
            "sae_checkpoint": str(Path(sae_checkpoint).expanduser().resolve()),
            "labels_source": str(Path(labels_source).expanduser().resolve()),
            "preprocessing": preprocessing,
        },
    )

    return {
        "out_dir": str(target_out_dir),
        "eval_summary_path": str(eval_summary_path),
        "eval_metrics_path": str(eval_metrics_path),
        "feature_stats_path": str(feature_stats_path),
        "feature_auc_path": str(feature_auc_path),
        "top_examples_path": str(top_examples_path),
        "feature_payload_path": str(feature_payload_path),
        "summary": summary,
        "feature_summary": feature_summary,
    }
