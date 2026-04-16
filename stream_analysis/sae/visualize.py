"""Plotting utilities for SAE metrics and diagnostics."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np


def _prepare_output(output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _to_float_array(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    return np.asarray(matrix, dtype=float)


def plot_training_metrics(metrics_csv: str | Path, output_path: str | Path) -> Path:
    """Plot train / val reconstruction curves from ``metrics.csv``."""

    steps = []
    train_recon = []
    val_recon = []
    with Path(metrics_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            steps.append(int(float(row["step"])))
            train_recon.append(float(row["train_recon_mse"]))
            val_recon.append(float(row["val_recon_mse"]))

    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.plot(steps, train_recon, label="train_recon_mse")
    axis.plot(steps, val_recon, label="val_recon_mse")
    axis.set_xlabel("step")
    axis.set_ylabel("MSE")
    axis.set_title("SAE reconstruction error")
    axis.grid(alpha=0.3)
    axis.legend()

    target = _prepare_output(output_path)
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return target


def plot_latent_firing_rates(records: Sequence[Mapping[str, float]], output_path: str | Path) -> Path:
    """Plot a simple firing-rate bar chart from latent-summary records."""

    x = [int(record["latent_index"]) for record in records]
    y = [float(record["firing_rate"]) for record in records]

    figure, axis = plt.subplots(figsize=(max(8.0, 0.25 * len(x)), 4.5))
    axis.bar(x, y)
    axis.set_xlabel("latent index")
    axis.set_ylabel("firing rate")
    axis.set_title("SAE latent firing rates")
    axis.grid(axis="y", alpha=0.3)

    target = _prepare_output(output_path)
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return target


def plot_recon_vs_k(
    records: Sequence[Mapping[str, object]],
    output_path: str | Path,
    *,
    metric_key: str = "normalized_recon_mse",
    title: str = "Reconstruction vs k",
) -> Path:
    """Plot a scalar metric against SAE sparsity ``k``."""

    sorted_records = sorted(records, key=lambda row: float(row["k"]))
    x = [float(row["k"]) for row in sorted_records]
    y = [float(row[metric_key]) for row in sorted_records]

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(x, y, marker="o")
    axis.set_xlabel("k")
    axis.set_ylabel(metric_key)
    axis.set_title(title)
    axis.grid(alpha=0.3)

    target = _prepare_output(output_path)
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return target


def plot_dead_latents_vs_k(
    records: Sequence[Mapping[str, object]],
    output_path: str | Path,
    *,
    title: str = "Dead Latent Fraction vs k",
) -> Path:
    """Plot dead latent fraction against SAE sparsity ``k``."""

    sorted_records = sorted(records, key=lambda row: float(row["k"]))
    x = [float(row["k"]) for row in sorted_records]
    y = [float(row["dead_latent_frac"]) for row in sorted_records]

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(x, y, marker="o")
    axis.set_xlabel("k")
    axis.set_ylabel("dead_latent_frac")
    axis.set_title(title)
    axis.grid(alpha=0.3)

    target = _prepare_output(output_path)
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return target


def plot_layerwise_selective_feature_counts(
    records: Sequence[Mapping[str, object]],
    output_path: str | Path,
    *,
    title: str = "Selective Feature Counts by Layer",
) -> Path:
    """Plot mem-selective and nonmem-selective counts by layer."""

    sorted_records = sorted(records, key=lambda row: int(row["layer_idx"]))
    layers = [int(row["layer_idx"]) for row in sorted_records]
    mem_counts = [float(row["mem_selective_feature_count"]) for row in sorted_records]
    nonmem_counts = [float(row["nonmem_selective_feature_count"]) for row in sorted_records]

    x = np.arange(len(layers))
    width = 0.38

    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.bar(x - width / 2, mem_counts, width=width, label="mem-selective")
    axis.bar(x + width / 2, nonmem_counts, width=width, label="nonmem-selective")
    axis.set_xticks(x)
    axis.set_xticklabels([str(layer) for layer in layers])
    axis.set_xlabel("layer")
    axis.set_ylabel("feature count")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.3)
    axis.legend()

    target = _prepare_output(output_path)
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return target


def _plot_heatmap(
    matrix: Sequence[Sequence[float]],
    output_path: str | Path,
    *,
    title: str,
    colorbar_label: str,
) -> Path:
    array = _to_float_array(matrix)
    if array.ndim != 2:
        raise ValueError(f"heatmap matrix must be 2D, got shape {array.shape}.")

    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(array, aspect="auto", interpolation="nearest")
    axis.set_xlabel("latent j")
    axis.set_ylabel("latent i")
    axis.set_title(title)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(colorbar_label)

    target = _prepare_output(output_path)
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return target


def plot_decoder_overlap_heatmap(
    matrix: Sequence[Sequence[float]],
    output_path: str | Path,
    *,
    title: str = "Decoder Cosine Overlap",
) -> Path:
    """Plot a decoder-overlap heatmap."""

    return _plot_heatmap(matrix, output_path, title=title, colorbar_label="cosine overlap")


def plot_coactivation_heatmap(
    matrix: Sequence[Sequence[float]],
    output_path: str | Path,
    *,
    title: str = "Latent Coactivation Frequency",
) -> Path:
    """Plot a latent coactivation heatmap."""

    return _plot_heatmap(matrix, output_path, title=title, colorbar_label="coactivation frequency")


def plot_model_comparison_by_layer(
    records: Sequence[Mapping[str, object]],
    output_path: str | Path,
    *,
    metric_key: str = "normalized_recon_mse",
    title: str | None = None,
) -> Path:
    """Plot one scalar metric by layer for each model type."""

    grouped: dict[str, list[Mapping[str, object]]] = {}
    for record in records:
        model_type = str(record["model_type"])
        grouped.setdefault(model_type, []).append(record)

    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    for model_type, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: int(row["layer_idx"]))
        x = [int(row["layer_idx"]) for row in ordered]
        y = [float(row[metric_key]) for row in ordered]
        axis.plot(x, y, marker="o", label=model_type)

    axis.set_xlabel("layer")
    axis.set_ylabel(metric_key)
    axis.set_title(title or f"{metric_key} by layer and model")
    axis.grid(alpha=0.3)
    axis.legend()

    target = _prepare_output(output_path)
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return target


def plot_feature_effect_ranking(
    records: Sequence[Mapping[str, object]],
    output_path: str | Path,
    *,
    metric_key: str = "delta_logprob_all",
    title: str = "Feature Intervention Effects",
) -> Path:
    """Plot per-feature intervention effects ranked by one metric."""

    ordered = sorted(records, key=lambda row: float(row[metric_key]), reverse=True)
    labels = [
        str(row.get("feature_id", row.get("feature_ids", idx)))
        for idx, row in enumerate(ordered)
    ]
    values = [float(row[metric_key]) for row in ordered]
    y = np.arange(len(values))

    figure, axis = plt.subplots(figsize=(max(7.0, 0.35 * len(values) + 4.0), 5.0))
    axis.barh(y, values)
    axis.set_yticks(y)
    axis.set_yticklabels(labels)
    axis.invert_yaxis()
    axis.set_xlabel(metric_key)
    axis.set_ylabel("feature")
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.3)

    target = _prepare_output(output_path)
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return target


def plot_mem_vs_nonmem_effects(
    records: Sequence[Mapping[str, object]],
    output_path: str | Path,
    *,
    x_key: str = "delta_logprob_nonmem",
    y_key: str = "delta_logprob_mem",
    title: str = "Mem vs Nonmem Intervention Effects",
) -> Path:
    """Scatter mem and nonmem intervention effects for each feature."""

    x = [float(row[x_key]) for row in records]
    y = [float(row[y_key]) for row in records]

    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    axis.scatter(x, y, alpha=0.8)
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_xlabel(x_key)
    axis.set_ylabel(y_key)
    axis.set_title(title)
    axis.grid(alpha=0.3)

    target = _prepare_output(output_path)
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return target


def plot_layerwise_intervention_summary(
    records: Sequence[Mapping[str, object]],
    output_path: str | Path,
    *,
    metric_key: str = "delta_logprob_all",
    title: str = "Layer-wise Intervention Summary",
) -> Path:
    """Plot mean intervention effect per layer."""

    grouped: dict[int, list[float]] = {}
    for row in records:
        grouped.setdefault(int(row["layer_idx"]), []).append(float(row[metric_key]))
    layers = sorted(grouped)
    values = [float(np.mean(grouped[layer])) for layer in layers]

    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    axis.plot(layers, values, marker="o")
    axis.set_xlabel("layer")
    axis.set_ylabel(metric_key)
    axis.set_title(title)
    axis.grid(alpha=0.3)

    target = _prepare_output(output_path)
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return target


def _checkpoint_sort_value(value: object) -> tuple[int, str]:
    try:
        return (0, str(int(float(value))))
    except Exception:
        return (1, str(value))


def plot_checkpoint_effect_curve(
    records: Sequence[Mapping[str, object]],
    output_path: str | Path,
    *,
    metric_key: str = "delta_logprob_all",
    title: str = "Checkpoint-wise Effect Curve",
) -> Path:
    """Plot mean intervention effect across checkpoints."""

    grouped: dict[str, list[float]] = {}
    for row in records:
        grouped.setdefault(str(row["checkpoint_step"]), []).append(float(row[metric_key]))

    ordered_steps = sorted(grouped, key=_checkpoint_sort_value)
    x = np.arange(len(ordered_steps))
    y = [float(np.mean(grouped[step])) for step in ordered_steps]

    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.plot(x, y, marker="o")
    axis.set_xticks(x)
    axis.set_xticklabels(ordered_steps, rotation=30, ha="right")
    axis.set_xlabel("checkpoint_step")
    axis.set_ylabel(metric_key)
    axis.set_title(title)
    axis.grid(alpha=0.3)

    target = _prepare_output(output_path)
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return target


def plot_model_intervention_comparison(
    records: Sequence[Mapping[str, object]],
    output_path: str | Path,
    *,
    metric_key: str = "delta_logprob_all",
    title: str = "Model Intervention Comparison",
) -> Path:
    """Plot mean intervention effect by layer for each model type."""

    grouped: dict[str, dict[int, list[float]]] = {}
    for row in records:
        model_type = str(row["model_type"])
        layer_idx = int(row["layer_idx"])
        grouped.setdefault(model_type, {}).setdefault(layer_idx, []).append(float(row[metric_key]))

    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    for model_type, layer_map in sorted(grouped.items()):
        layers = sorted(layer_map)
        values = [float(np.mean(layer_map[layer])) for layer in layers]
        axis.plot(layers, values, marker="o", label=model_type)

    axis.set_xlabel("layer")
    axis.set_ylabel(metric_key)
    axis.set_title(title)
    axis.grid(alpha=0.3)
    axis.legend()

    target = _prepare_output(output_path)
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return target
