"""End-to-end SAE intervention experiments and batch studies."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch

from scripts.evaluate import load_checkpoint

from .analysis import (
    load_feature_stats_rows,
    load_mem_labels,
    run_sae_mem_analysis,
    select_top_feature_ids,
)
from .eval import load_sae_checkpoint
from .patching import build_sae_site_override, extract_rows_from_cache, site_cache_key
from .utils import (
    default_sae_analysis_dir,
    default_sae_intervention_dir,
    ensure_dir,
    format_checkpoint_step,
    read_json,
    resolve_device,
    safe_float,
    sanitize_component,
    write_csv_rows,
    write_json,
)
from .visualize import (
    plot_checkpoint_effect_curve,
    plot_feature_effect_ranking,
    plot_layerwise_intervention_summary,
    plot_mem_vs_nonmem_effects,
    plot_model_intervention_comparison,
)

logger = logging.getLogger("sae.experiment")


@dataclass
class SAERuntimeContext:
    """Minimal runtime metadata derived from a trained SAE checkpoint."""

    sae_checkpoint_path: str
    source_checkpoint_path: str
    activation_dir: str
    model_type: str
    checkpoint_step: object
    layer_idx: int
    site: str
    preprocessing: str


def parse_feature_ids(value: str | Sequence[int] | None) -> List[int]:
    """Parse feature ids from CLI-style strings or existing sequences."""

    if value is None:
        return []
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
        return [int(piece) for piece in pieces]
    return [int(item) for item in value]


def _resolve_sae_runtime_context(
    sae_checkpoint_path: str | Path,
    sae_payload: Mapping[str, Any],
    *,
    activation_dir: str | Path | None = None,
) -> SAERuntimeContext:
    activation_metadata = sae_payload.get("activation_metadata", {})
    merged_meta: Dict[str, Any] = {}
    if isinstance(activation_metadata, Mapping):
        merged_meta.update(dict(activation_metadata))

    payload_activation_dir = sae_payload.get("activation_dir")
    resolved_activation_dir = (
        Path(activation_dir).expanduser().resolve()
        if activation_dir is not None
        else (
            Path(str(payload_activation_dir)).expanduser().resolve()
            if isinstance(payload_activation_dir, str) and payload_activation_dir.strip()
            else None
        )
    )
    if resolved_activation_dir is not None:
        meta_path = resolved_activation_dir / "meta.json"
        if meta_path.is_file():
            activation_meta = read_json(meta_path)
            if isinstance(activation_meta, Mapping):
                merged_meta = dict(activation_meta) | merged_meta

    checkpoint_path = merged_meta.get("checkpoint_path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
        raise KeyError("SAE checkpoint is missing activation_metadata.checkpoint_path.")

    train_config = sae_payload.get("train_config", {})
    preprocessing = "none"
    if isinstance(train_config, Mapping):
        preprocessing = str(train_config.get("preprocessing", "none"))

    return SAERuntimeContext(
        sae_checkpoint_path=str(Path(sae_checkpoint_path).expanduser().resolve()),
        source_checkpoint_path=str(Path(checkpoint_path).expanduser().resolve()),
        activation_dir="" if resolved_activation_dir is None else str(resolved_activation_dir),
        model_type=str(merged_meta.get("model_type", "unknown")),
        checkpoint_step=merged_meta.get("checkpoint_step"),
        layer_idx=int(merged_meta.get("layer_idx", 0)),
        site=str(merged_meta.get("site", "input")),
        preprocessing=preprocessing,
    )


def _validate_runtime_context(
    context: SAERuntimeContext,
    *,
    model_type: str | None = None,
    checkpoint_path: str | Path | None = None,
    layer_idx: int | None = None,
    site: str | None = None,
) -> None:
    if model_type is not None and context.model_type != model_type:
        raise ValueError(f"SAE model_type={context.model_type!r} does not match requested model_type={model_type!r}.")
    if checkpoint_path is not None:
        resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
        if resolved_checkpoint != Path(context.source_checkpoint_path):
            raise ValueError(
                "Requested checkpoint does not match the checkpoint used to extract SAE activations. "
                f"requested={resolved_checkpoint}, sae_source={context.source_checkpoint_path}"
            )
    if layer_idx is not None and context.layer_idx != int(layer_idx):
        raise ValueError(f"SAE layer_idx={context.layer_idx} does not match requested layer_idx={layer_idx}.")
    if site is not None and context.site != str(site).strip().lower():
        raise ValueError(f"SAE site={context.site!r} does not match requested site={site!r}.")


def _require_labels(label_artifact) -> torch.Tensor:
    labels = label_artifact.labels
    if labels is None:
        raise ValueError("Intervention experiments require labels in the labels_source artifact.")
    return labels


def _select_gold_tokens(labels: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
    batch_index = torch.arange(labels.shape[0], dtype=torch.long)
    return labels[batch_index, target_positions]


def compute_target_token_metrics(
    logits: torch.Tensor,
    *,
    target_positions: torch.Tensor,
    gold_token_ids: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute per-sample target-position metrics from logits."""

    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B, T, V], got {tuple(logits.shape)}.")
    if target_positions.ndim != 1 or gold_token_ids.ndim != 1:
        raise ValueError("target_positions and gold_token_ids must be 1D tensors.")
    if int(logits.shape[0]) != int(target_positions.shape[0]) or int(logits.shape[0]) != int(gold_token_ids.shape[0]):
        raise ValueError("logits, target_positions, and gold_token_ids must share batch size.")

    batch_index = torch.arange(logits.shape[0], device=logits.device)
    target_positions = target_positions.to(device=logits.device, dtype=torch.long)
    gold_token_ids = gold_token_ids.to(device=logits.device, dtype=torch.long)

    target_logits = logits[batch_index, target_positions]
    gold_logits = target_logits[batch_index, gold_token_ids]
    log_probs = torch.log_softmax(target_logits, dim=-1)
    gold_logprob = log_probs[batch_index, gold_token_ids]

    competitor_logits = target_logits.clone()
    competitor_logits[batch_index, gold_token_ids] = float("-inf")
    max_other_logits = competitor_logits.max(dim=-1).values
    logit_diff = gold_logits - max_other_logits

    return {
        "gold_logit": gold_logits,
        "gold_logprob": gold_logprob,
        "logit_diff": logit_diff,
    }


def _mean_for_mask(values: Sequence[float], mask: Sequence[bool]) -> float:
    selected = [float(value) for value, keep in zip(values, mask) if keep]
    if not selected:
        return float("nan")
    return float(sum(selected) / len(selected))


def _summarize_effects(
    sample_rows: Sequence[Mapping[str, object]],
    *,
    mode: str,
) -> Dict[str, float]:
    is_mem = [bool(row["is_mem"]) for row in sample_rows]
    is_nonmem = [not flag for flag in is_mem]

    delta_logprob = [safe_float(row["delta_logprob"]) for row in sample_rows]
    delta_logit_diff = [safe_float(row["delta_logit_diff"]) for row in sample_rows]
    summary = {
        "delta_logprob_all": _mean_for_mask(delta_logprob, [True] * len(delta_logprob)),
        "delta_logprob_mem": _mean_for_mask(delta_logprob, is_mem),
        "delta_logprob_nonmem": _mean_for_mask(delta_logprob, is_nonmem),
        "delta_logit_diff_all": _mean_for_mask(delta_logit_diff, [True] * len(delta_logit_diff)),
        "delta_logit_diff_mem": _mean_for_mask(delta_logit_diff, is_mem),
        "delta_logit_diff_nonmem": _mean_for_mask(delta_logit_diff, is_nonmem),
    }
    summary["mem_specific_logprob"] = summary["delta_logprob_mem"] - summary["delta_logprob_nonmem"]
    summary["mem_specific_logit_diff"] = summary["delta_logit_diff_mem"] - summary["delta_logit_diff_nonmem"]

    if mode == "patch":
        recovery_logprob = [safe_float(row["recovery_logprob"]) for row in sample_rows]
        recovery_logit_diff = [safe_float(row["recovery_logit_diff"]) for row in sample_rows]
        summary.update(
            {
                "recovery_logprob_all": _mean_for_mask(recovery_logprob, [True] * len(recovery_logprob)),
                "recovery_logprob_mem": _mean_for_mask(recovery_logprob, is_mem),
                "recovery_logprob_nonmem": _mean_for_mask(recovery_logprob, is_nonmem),
                "recovery_logit_diff_all": _mean_for_mask(recovery_logit_diff, [True] * len(recovery_logit_diff)),
                "recovery_logit_diff_mem": _mean_for_mask(recovery_logit_diff, is_mem),
                "recovery_logit_diff_nonmem": _mean_for_mask(recovery_logit_diff, is_nonmem),
            }
        )
    return summary


def _to_subgroup_rows(feature_rows: Sequence[Mapping[str, object]], subgroup: str) -> List[Dict[str, object]]:
    suffix = "mem" if subgroup == "mem" else "nonmem"
    rows: List[Dict[str, object]] = []
    for row in feature_rows:
        rows.append(
            {
                "feature_id": row["feature_id"],
                "feature_ids": row["feature_ids"],
                "mode": row["mode"],
                "scale_factor": row["scale_factor"],
                "layer_idx": row["layer_idx"],
                "site": row["site"],
                "model_type": row["model_type"],
                "checkpoint_step": row["checkpoint_step"],
                "delta_logprob": row[f"delta_logprob_{suffix}"],
                "delta_logit_diff": row[f"delta_logit_diff_{suffix}"],
                "mem_specific_logprob": row["mem_specific_logprob"],
                "mem_specific_logit_diff": row["mem_specific_logit_diff"],
            }
        )
    return rows


def _maybe_run_mem_analysis(
    *,
    context: SAERuntimeContext,
    sae_checkpoint_path: str | Path,
    labels_source: str | Path,
    batch_size: int,
    device: str | torch.device,
) -> Path:
    analysis_dir = default_sae_analysis_dir(
        context.model_type,
        context.checkpoint_step,
        context.layer_idx,
        context.site,
    )
    feature_stats_path = analysis_dir / "feature_stats.csv"
    if feature_stats_path.is_file():
        return analysis_dir
    if not context.activation_dir:
        raise FileNotFoundError(
            "feature_stats.csv is missing and the SAE checkpoint does not record an activation_dir to rebuild analysis outputs."
        )
    logger.info("feature_stats.csv not found; running mem analysis at %s", analysis_dir)
    run_sae_mem_analysis(
        sae_checkpoint_path,
        context.activation_dir,
        labels_source,
        batch_size=batch_size,
        device=device,
        out_dir=analysis_dir,
    )
    return analysis_dir


def select_features_for_sweep(
    *,
    selection: str,
    topn: int,
    manual_feature_ids: Sequence[int] | None,
    sae_model_n_latents: int,
    context: SAERuntimeContext,
    sae_checkpoint_path: str | Path,
    labels_source: str | Path,
    batch_size: int,
    device: str | torch.device,
) -> List[int]:
    """Resolve the feature ids used by a sweep."""

    normalized_selection = str(selection).strip().lower()
    if normalized_selection == "manual":
        feature_ids = parse_feature_ids(manual_feature_ids)
        if not feature_ids:
            raise ValueError("manual selection requires at least one feature id.")
        for feature_id in feature_ids:
            if feature_id < 0 or feature_id >= sae_model_n_latents:
                raise IndexError(
                    f"feature_id {feature_id} is out of range for n_latents={sae_model_n_latents}."
                )
        return [feature_id for feature_id in feature_ids]

    if normalized_selection not in {"top_mem", "top_nonmem"}:
        raise ValueError("selection must be one of: top_mem, top_nonmem, manual.")

    analysis_dir = _maybe_run_mem_analysis(
        context=context,
        sae_checkpoint_path=sae_checkpoint_path,
        labels_source=labels_source,
        batch_size=batch_size,
        device=device,
    )
    rows = load_feature_stats_rows(analysis_dir)
    selected = select_top_feature_ids(rows, selection=normalized_selection, topn=topn)
    for feature_id in selected:
        if feature_id < 0 or feature_id >= sae_model_n_latents:
            raise IndexError(
                f"feature_id {feature_id} selected from analysis outputs is out of range for n_latents={sae_model_n_latents}."
            )
    return selected


def _save_intervention_bundle(
    *,
    feature_rows: Sequence[Mapping[str, object]],
    sample_rows: Sequence[Mapping[str, object]],
    out_dir: str | Path,
    summary_payload: Mapping[str, object],
) -> Dict[str, str]:
    output_dir = ensure_dir(out_dir)
    feature_effects_path = write_csv_rows(output_dir / "feature_effects.csv", list(feature_rows[0].keys()) if feature_rows else [], feature_rows)
    feature_effects_mem_path = write_csv_rows(
        output_dir / "feature_effects_mem.csv",
        ["feature_id", "feature_ids", "mode", "scale_factor", "layer_idx", "site", "model_type", "checkpoint_step", "delta_logprob", "delta_logit_diff", "mem_specific_logprob", "mem_specific_logit_diff"],
        _to_subgroup_rows(feature_rows, "mem"),
    )
    feature_effects_nonmem_path = write_csv_rows(
        output_dir / "feature_effects_nonmem.csv",
        ["feature_id", "feature_ids", "mode", "scale_factor", "layer_idx", "site", "model_type", "checkpoint_step", "delta_logprob", "delta_logit_diff", "mem_specific_logprob", "mem_specific_logit_diff"],
        _to_subgroup_rows(feature_rows, "nonmem"),
    )
    sample_rows_path = write_csv_rows(output_dir / "per_sample_effects.csv", list(sample_rows[0].keys()) if sample_rows else [], sample_rows)
    summary_path = write_json(dict(summary_payload), output_dir / "summary.json")
    return {
        "feature_effects": str(feature_effects_path),
        "feature_effects_mem": str(feature_effects_mem_path),
        "feature_effects_nonmem": str(feature_effects_nonmem_path),
        "per_sample_effects": str(sample_rows_path),
        "summary": str(summary_path),
    }


def _save_intervention_figures(
    feature_rows: Sequence[Mapping[str, object]],
    *,
    out_dir: str | Path,
    prefix: str = "",
) -> Dict[str, str]:
    figures_dir = ensure_dir(Path(out_dir) / "figures")
    stem_prefix = "" if not prefix else f"{sanitize_component(prefix)}_"
    top_effects = plot_feature_effect_ranking(
        feature_rows,
        figures_dir / f"{stem_prefix}top_feature_effects.png",
    )
    mem_vs_nonmem = plot_mem_vs_nonmem_effects(
        feature_rows,
        figures_dir / f"{stem_prefix}mem_vs_nonmem_effects.png",
    )
    return {
        "top_feature_effects": str(top_effects),
        "mem_vs_nonmem_effects": str(mem_vs_nonmem),
    }


def run_sae_intervention_experiment(
    *,
    model_type: str,
    checkpoint_path: str | Path,
    sae_checkpoint_path: str | Path,
    labels_source: str | Path,
    layer_idx: int,
    site: str,
    mode: str,
    feature_ids: Sequence[int],
    dataset_split: str = "val",
    batch_size: int = 32,
    device: str | torch.device = "auto",
    out_dir: str | Path | None = None,
    activation_dir: str | Path | None = None,
    scale_factor: float = 1.0,
    preserve_error: bool = True,
    donor_labels_source: str | Path | None = None,
    write_outputs: bool = True,
) -> Dict[str, object]:
    """Run one SAE feature intervention experiment over an analysis set."""

    torch_device = resolve_device(str(device)) if not isinstance(device, torch.device) else device
    sae_model, _, sae_payload = load_sae_checkpoint(sae_checkpoint_path, device=torch_device)
    context = _resolve_sae_runtime_context(sae_checkpoint_path, sae_payload, activation_dir=activation_dir)
    _validate_runtime_context(
        context,
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        layer_idx=layer_idx,
        site=site,
    )

    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"zero", "keep_only", "scale", "patch"}:
        raise ValueError("mode must be one of: zero, keep_only, scale, patch.")

    source_model, _, checkpoint_payload = load_checkpoint(checkpoint_path, device=torch_device)
    label_artifact = load_mem_labels(labels_source)
    labels = _require_labels(label_artifact)
    donor_artifact = None
    donor_labels = None
    if normalized_mode == "patch":
        if donor_labels_source is None:
            raise ValueError("patch mode requires donor_labels_source.")
        donor_artifact = load_mem_labels(donor_labels_source)
        donor_labels = _require_labels(donor_artifact)
        if donor_artifact.num_samples != label_artifact.num_samples:
            raise ValueError(
                "patch mode requires donor and corrupt analysis sets with the same number of samples. "
                f"Got donor={donor_artifact.num_samples}, corrupt={label_artifact.num_samples}."
            )
        if not torch.equal(donor_artifact.target_positions, label_artifact.target_positions):
            raise ValueError("patch mode requires donor and corrupt target_positions to match.")
        donor_gold = _select_gold_tokens(donor_labels, donor_artifact.target_positions)
        corrupt_gold = _select_gold_tokens(labels, label_artifact.target_positions)
        if not torch.equal(donor_gold, corrupt_gold):
            raise ValueError("patch mode requires donor and corrupt gold target tokens to match.")

    cache_key = site_cache_key(layer_idx, site)
    feature_id_list = [int(feature_id) for feature_id in feature_ids]
    if not feature_id_list:
        raise ValueError("feature_ids must be non-empty.")

    feature_descriptor = ",".join(str(feature_id) for feature_id in feature_id_list)
    sample_rows: List[Dict[str, object]] = []
    num_samples = label_artifact.num_samples
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_input_ids = label_artifact.input_ids[start:end].to(device=torch_device, dtype=torch.long)
        batch_labels = labels[start:end]
        batch_target_positions = label_artifact.target_positions[start:end]
        batch_gold = _select_gold_tokens(batch_labels, batch_target_positions)

        with torch.no_grad():
            if normalized_mode == "patch":
                assert donor_artifact is not None and donor_labels is not None
                donor_input_ids = donor_artifact.input_ids[start:end].to(device=torch_device, dtype=torch.long)
                donor_target_positions = donor_artifact.target_positions[start:end]
                clean_outputs = source_model(
                    donor_input_ids,
                    return_intermediates=True,
                    return_cache=True,
                )
                corrupt_outputs = source_model(batch_input_ids)
                donor_rows = extract_rows_from_cache(
                    clean_outputs["intermediates"],
                    cache_key=cache_key,
                    target_positions=donor_target_positions,
                )
                override = build_sae_site_override(
                    sae_model,
                    target_positions=batch_target_positions,
                    mode=normalized_mode,
                    feature_ids=feature_id_list,
                    preprocessing=context.preprocessing,
                    scale_factor=scale_factor,
                    donor_rows=donor_rows,
                    preserve_error=preserve_error,
                )
                patched_outputs = source_model(
                    batch_input_ids,
                    activation_overrides={cache_key: override},
                )
                clean_metrics = compute_target_token_metrics(
                    clean_outputs["logits"],
                    target_positions=donor_target_positions,
                    gold_token_ids=batch_gold,
                )
                corrupt_metrics = compute_target_token_metrics(
                    corrupt_outputs["logits"],
                    target_positions=batch_target_positions,
                    gold_token_ids=batch_gold,
                )
                patched_metrics = compute_target_token_metrics(
                    patched_outputs["logits"],
                    target_positions=batch_target_positions,
                    gold_token_ids=batch_gold,
                )
            else:
                baseline_outputs = source_model(batch_input_ids)
                override = build_sae_site_override(
                    sae_model,
                    target_positions=batch_target_positions,
                    mode=normalized_mode,
                    feature_ids=feature_id_list,
                    preprocessing=context.preprocessing,
                    scale_factor=scale_factor,
                    preserve_error=preserve_error,
                )
                edited_outputs = source_model(
                    batch_input_ids,
                    activation_overrides={cache_key: override},
                )
                baseline_metrics = compute_target_token_metrics(
                    baseline_outputs["logits"],
                    target_positions=batch_target_positions,
                    gold_token_ids=batch_gold,
                )
                edited_metrics = compute_target_token_metrics(
                    edited_outputs["logits"],
                    target_positions=batch_target_positions,
                    gold_token_ids=batch_gold,
                )

        for local_index, sample_index in enumerate(range(start, end)):
            base_row: Dict[str, object] = {
                "sample_id": label_artifact.sample_ids[sample_index],
                "group_label": label_artifact.group_labels[sample_index],
                "is_mem": bool(label_artifact.mem_labels[sample_index].item()),
                "target_position": int(label_artifact.target_positions[sample_index].item()),
                "gold_token_id": int(batch_gold[local_index].item()),
                "feature_ids": feature_descriptor,
                "mode": normalized_mode,
                "scale_factor": float(scale_factor),
                "layer_idx": layer_idx,
                "site": site,
                "model_type": model_type,
                "checkpoint_step": checkpoint_payload.get("step", context.checkpoint_step),
            }
            if normalized_mode == "patch":
                clean_logprob = float(clean_metrics["gold_logprob"][local_index].item())
                corrupt_logprob = float(corrupt_metrics["gold_logprob"][local_index].item())
                patched_logprob = float(patched_metrics["gold_logprob"][local_index].item())
                clean_logit_diff = float(clean_metrics["logit_diff"][local_index].item())
                corrupt_logit_diff = float(corrupt_metrics["logit_diff"][local_index].item())
                patched_logit_diff = float(patched_metrics["logit_diff"][local_index].item())
                delta_logprob = patched_logprob - corrupt_logprob
                delta_logit_diff = patched_logit_diff - corrupt_logit_diff
                denom_logprob = clean_logprob - corrupt_logprob
                denom_logit_diff = clean_logit_diff - corrupt_logit_diff
                row = {
                    **base_row,
                    "clean_logprob": clean_logprob,
                    "corrupt_logprob": corrupt_logprob,
                    "patched_logprob": patched_logprob,
                    "delta_logprob": delta_logprob,
                    "recovery_logprob": float("nan") if abs(denom_logprob) <= 1e-12 else delta_logprob / denom_logprob,
                    "clean_logit_diff": clean_logit_diff,
                    "corrupt_logit_diff": corrupt_logit_diff,
                    "patched_logit_diff": patched_logit_diff,
                    "delta_logit_diff": delta_logit_diff,
                    "recovery_logit_diff": float("nan")
                    if abs(denom_logit_diff) <= 1e-12
                    else delta_logit_diff / denom_logit_diff,
                }
            else:
                baseline_logprob = float(baseline_metrics["gold_logprob"][local_index].item())
                edited_logprob = float(edited_metrics["gold_logprob"][local_index].item())
                baseline_logit_diff = float(baseline_metrics["logit_diff"][local_index].item())
                edited_logit_diff = float(edited_metrics["logit_diff"][local_index].item())
                row = {
                    **base_row,
                    "baseline_logprob": baseline_logprob,
                    "edited_logprob": edited_logprob,
                    "delta_logprob": edited_logprob - baseline_logprob,
                    "baseline_logit_diff": baseline_logit_diff,
                    "edited_logit_diff": edited_logit_diff,
                    "delta_logit_diff": edited_logit_diff - baseline_logit_diff,
                }
            sample_rows.append(row)

    effect_summary = _summarize_effects(sample_rows, mode=normalized_mode)
    feature_row = {
        "feature_id": feature_id_list[0] if len(feature_id_list) == 1 else -1,
        "feature_ids": feature_descriptor,
        "num_features": len(feature_id_list),
        "mode": normalized_mode,
        "scale_factor": float(scale_factor),
        "preserve_error": bool(preserve_error),
        "dataset_split": dataset_split,
        "layer_idx": layer_idx,
        "site": site,
        "model_type": model_type,
        "checkpoint_step": checkpoint_payload.get("step", context.checkpoint_step),
        "labels_source": str(Path(labels_source).expanduser().resolve()),
        "sae_checkpoint": context.sae_checkpoint_path,
        **effect_summary,
    }

    summary_payload = {
        "summary": feature_row,
        "num_samples": len(sample_rows),
        "num_mem_samples": int(sum(1 for row in sample_rows if bool(row["is_mem"]))),
        "num_nonmem_samples": int(sum(1 for row in sample_rows if not bool(row["is_mem"]))),
        "cache_key": cache_key,
        "source_checkpoint": context.source_checkpoint_path,
        "activation_dir": context.activation_dir,
        "preprocessing": context.preprocessing,
        "logit_diff_definition": "gold_logit - max_non_gold_logit",
    }

    saved_paths: Dict[str, str] = {}
    if write_outputs:
        target_out_dir = (
            Path(out_dir).expanduser().resolve()
            if out_dir is not None
            else default_sae_intervention_dir(model_type, checkpoint_payload.get("step", context.checkpoint_step), layer_idx, site)
        )
        saved_paths = _save_intervention_bundle(
            feature_rows=[feature_row],
            sample_rows=sample_rows,
            out_dir=target_out_dir,
            summary_payload=summary_payload,
        )
        saved_paths.update(_save_intervention_figures([feature_row], out_dir=target_out_dir))

    return {
        "context": context,
        "feature_rows": [feature_row],
        "sample_rows": sample_rows,
        "summary": summary_payload,
        "saved_paths": saved_paths,
    }


def run_sae_feature_sweep(
    *,
    model_type: str,
    checkpoint_path: str | Path,
    sae_checkpoint_path: str | Path,
    labels_source: str | Path,
    layer_idx: int,
    site: str,
    selection: str,
    topn: int,
    dataset_split: str = "val",
    batch_size: int = 32,
    device: str | torch.device = "auto",
    out_dir: str | Path | None = None,
    activation_dir: str | Path | None = None,
    manual_feature_ids: Sequence[int] | None = None,
    mode: str = "zero",
    scale_factor: float = 1.0,
    preserve_error: bool = True,
) -> Dict[str, object]:
    """Run a per-feature SAE intervention sweep."""

    torch_device = resolve_device(str(device)) if not isinstance(device, torch.device) else device
    sae_model, _, sae_payload = load_sae_checkpoint(sae_checkpoint_path, device=torch_device)
    context = _resolve_sae_runtime_context(sae_checkpoint_path, sae_payload, activation_dir=activation_dir)
    _validate_runtime_context(
        context,
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        layer_idx=layer_idx,
        site=site,
    )

    feature_ids = select_features_for_sweep(
        selection=selection,
        topn=topn,
        manual_feature_ids=manual_feature_ids,
        sae_model_n_latents=sae_model.n_latents,
        context=context,
        sae_checkpoint_path=sae_checkpoint_path,
        labels_source=labels_source,
        batch_size=batch_size,
        device=device,
    )

    combined_feature_rows: List[Dict[str, object]] = []
    combined_sample_rows: List[Dict[str, object]] = []
    for feature_id in feature_ids:
        result = run_sae_intervention_experiment(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            sae_checkpoint_path=sae_checkpoint_path,
            labels_source=labels_source,
            layer_idx=layer_idx,
            site=site,
            mode=mode,
            feature_ids=[feature_id],
            dataset_split=dataset_split,
            batch_size=batch_size,
            device=device,
            out_dir=None,
            activation_dir=activation_dir,
            scale_factor=scale_factor,
            preserve_error=preserve_error,
            write_outputs=False,
        )
        combined_feature_rows.extend(result["feature_rows"])
        combined_sample_rows.extend(result["sample_rows"])

    checkpoint_payload = load_checkpoint(checkpoint_path, device=torch_device)[2]
    target_out_dir = (
        Path(out_dir).expanduser().resolve()
        if out_dir is not None
        else default_sae_intervention_dir(model_type, checkpoint_payload.get("step", context.checkpoint_step), layer_idx, site)
    )
    summary_payload = {
        "selection": selection,
        "topn": topn,
        "mode": mode,
        "num_features": len(combined_feature_rows),
        "model_type": model_type,
        "checkpoint_step": checkpoint_payload.get("step", context.checkpoint_step),
        "layer_idx": layer_idx,
        "site": site,
    }
    saved_paths = _save_intervention_bundle(
        feature_rows=combined_feature_rows,
        sample_rows=combined_sample_rows,
        out_dir=target_out_dir,
        summary_payload=summary_payload,
    )
    saved_paths.update(_save_intervention_figures(combined_feature_rows, out_dir=target_out_dir))

    return {
        "feature_rows": combined_feature_rows,
        "sample_rows": combined_sample_rows,
        "summary": summary_payload,
        "saved_paths": saved_paths,
    }


def run_sae_checkpoint_study(
    *,
    model_type: str | Sequence[str],
    layers: Sequence[int],
    sites: Sequence[str],
    sae_root: str | Path,
    checkpoints: Sequence[str | int],
    labels_source: str | Path,
    out_dir: str | Path,
    batch_size: int = 32,
    device: str | torch.device = "auto",
    selection: str = "top_mem",
    topn: int = 1,
    mode: str = "zero",
    scale_factor: float = 1.0,
    preserve_error: bool = True,
) -> Dict[str, object]:
    """Run a checkpoint-wise intervention study over one SAE root."""

    rows: List[Dict[str, object]] = []
    model_types = [model_type] if isinstance(model_type, str) else list(model_type)
    for current_model_type in model_types:
        for checkpoint_step in checkpoints:
            for layer_idx in layers:
                for site in sites:
                    sae_checkpoint_path = (
                        Path(sae_root).expanduser().resolve()
                        / sanitize_component(current_model_type)
                        / f"step_{format_checkpoint_step(checkpoint_step)}"
                        / f"layer_{int(layer_idx)}"
                        / sanitize_component(site)
                        / "best.pt"
                    )
                    if not sae_checkpoint_path.is_file():
                        raise FileNotFoundError(f"SAE checkpoint not found for checkpoint study: {sae_checkpoint_path}")

                    _, _, sae_payload = load_sae_checkpoint(sae_checkpoint_path, device=resolve_device(str(device)))
                    context = _resolve_sae_runtime_context(sae_checkpoint_path, sae_payload)
                    result = run_sae_feature_sweep(
                        model_type=current_model_type,
                        checkpoint_path=context.source_checkpoint_path,
                        sae_checkpoint_path=sae_checkpoint_path,
                        labels_source=labels_source,
                        layer_idx=context.layer_idx,
                        site=context.site,
                        selection=selection,
                        topn=topn,
                        dataset_split="analysis",
                        batch_size=batch_size,
                        device=device,
                        out_dir=None,
                        activation_dir=context.activation_dir or None,
                        mode=mode,
                        scale_factor=scale_factor,
                        preserve_error=preserve_error,
                    )
                    rows.extend(result["feature_rows"])

    output_dir = ensure_dir(out_dir)
    feature_effects_path = write_csv_rows(
        output_dir / "feature_effects.csv",
        list(rows[0].keys()) if rows else [],
        rows,
    )
    summary_payload = {
        "model_type": model_types,
        "selection": selection,
        "topn": topn,
        "mode": mode,
        "checkpoints": [format_checkpoint_step(checkpoint) for checkpoint in checkpoints],
        "layers": [int(layer) for layer in layers],
        "sites": [str(site) for site in sites],
    }
    summary_path = write_json(summary_payload, output_dir / "summary.json")

    figures_dir = ensure_dir(output_dir / "figures")
    checkpoint_curve = plot_checkpoint_effect_curve(rows, figures_dir / "checkpoint_curves.png")
    layer_summary = plot_layerwise_intervention_summary(rows, figures_dir / "layerwise_intervention_summary.png")
    model_compare = plot_model_intervention_comparison(rows, figures_dir / "model_comparison.png")

    return {
        "feature_rows": rows,
        "saved_paths": {
            "feature_effects": str(feature_effects_path),
            "summary": str(summary_path),
            "checkpoint_curves": str(checkpoint_curve),
            "layerwise_intervention_summary": str(layer_summary),
            "model_comparison": str(model_compare),
        },
    }
