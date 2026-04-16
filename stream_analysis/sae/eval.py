#!/usr/bin/env python3
"""SAE evaluation utilities and CLI."""

from __future__ import annotations

import argparse
import heapq
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from stream_analysis.path_utils import resolve_project_path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .config import SAEConfig, SAEEvalConfig, resolve_preprocessing_mode
from .data import ActivationShardDataset
from .losses import compute_loss_dict
from .model import TopKSAE
from .utils import (
    configure_logging,
    default_sae_eval_dir,
    ensure_dir,
    finite_stats,
    format_top_pairs,
    maybe_tqdm,
    maybe_make_dataframe,
    read_json,
    resolve_device,
    safe_float,
    stage_progress,
    write_csv_rows,
    write_json,
)

logger = logging.getLogger("sae.eval")


def _unwrap_sae_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying SAE module when wrapped by DDP/accelerate."""

    return getattr(model, "module", model)


def load_sae_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[TopKSAE, SAEConfig, Dict[str, Any]]:
    """Load a serialized SAE checkpoint."""

    resolved = resolve_project_path(checkpoint_path)
    payload = torch.load(resolved, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"SAE checkpoint must be a mapping, got {type(payload).__name__}.")

    sae_config_payload = payload.get("sae_config")
    if not isinstance(sae_config_payload, Mapping):
        raise KeyError("SAE checkpoint is missing 'sae_config'.")

    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping):
        raise KeyError("SAE checkpoint is missing 'model_state'.")

    sae_config = SAEConfig.from_dict(dict(sae_config_payload))
    sae_config.device = str(device)
    model = TopKSAE(sae_config).to(device)
    model.load_state_dict(model_state)
    model.eval()
    return model, sae_config, dict(payload)


def _coerce_batch(batch: object, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(batch, (list, tuple)):
        if not batch:
            raise ValueError("Received an empty batch.")
        batch = batch[0]
    if not torch.is_tensor(batch):
        raise TypeError(f"Expected tensor batch, got {type(batch).__name__}.")
    if batch.ndim != 2:
        raise ValueError(f"Expected [batch, d_in] activations, got {tuple(batch.shape)}.")
    return batch.to(device=device, dtype=dtype)


class _ReconstructionAccumulator:
    """Running reconstruction summary over arbitrary batches."""

    def __init__(self, n_latents: int) -> None:
        self.n_latents = int(n_latents)
        self.total_samples = 0
        self.total_elements = 0
        self.total_sq_error = 0.0
        self.total_loss = 0.0
        self.total_aux = 0.0
        self.total_l0 = 0.0
        self.total_input_sum = 0.0
        self.total_input_sq_sum = 0.0
        self.fire_count = torch.zeros(self.n_latents, dtype=torch.long)

    def update(
        self,
        inputs: torch.Tensor,
        outputs: Mapping[str, torch.Tensor],
        *,
        use_auxk: bool,
        auxk_alpha: float,
        dead_threshold: float,
    ) -> None:
        losses = compute_loss_dict(
            inputs,
            outputs,
            use_auxk=use_auxk,
            auxk_alpha=auxk_alpha,
            dead_threshold=dead_threshold,
        )
        batch_size = int(inputs.shape[0])
        self.total_samples += batch_size
        self.total_elements += int(inputs.numel())
        self.total_sq_error += float(outputs["recon_error"].pow(2).sum().item())
        self.total_loss += float(losses["loss"].item()) * batch_size
        self.total_aux += float(losses["auxk_loss"].item()) * batch_size
        self.total_l0 += float(losses["avg_l0"].item()) * batch_size
        self.total_input_sum += float(inputs.sum().item())
        self.total_input_sq_sum += float(inputs.pow(2).sum().item())
        self.fire_count += (outputs["z"].detach().cpu().abs() > dead_threshold).sum(dim=0)

    def summary(
        self,
        *,
        accelerator: Any | None = None,
        device: torch.device | None = None,
    ) -> Dict[str, float]:
        total_samples = float(self.total_samples)
        total_elements = float(self.total_elements)
        total_sq_error = float(self.total_sq_error)
        total_loss = float(self.total_loss)
        total_aux = float(self.total_aux)
        total_l0 = float(self.total_l0)
        total_input_sum = float(self.total_input_sum)
        total_input_sq_sum = float(self.total_input_sq_sum)
        fire_count = self.fire_count

        if accelerator is not None and int(getattr(accelerator, "num_processes", 1)) > 1:
            reduce_device = accelerator.device if device is None else device
            totals = torch.tensor(
                [
                    total_samples,
                    total_elements,
                    total_sq_error,
                    total_loss,
                    total_aux,
                    total_l0,
                    total_input_sum,
                    total_input_sq_sum,
                ],
                device=reduce_device,
                dtype=torch.float64,
            )
            totals = accelerator.reduce(totals, reduction="sum")
            total_samples = float(totals[0].item())
            total_elements = float(totals[1].item())
            total_sq_error = float(totals[2].item())
            total_loss = float(totals[3].item())
            total_aux = float(totals[4].item())
            total_l0 = float(totals[5].item())
            total_input_sum = float(totals[6].item())
            total_input_sq_sum = float(totals[7].item())
            fire_count = accelerator.reduce(
                fire_count.to(device=reduce_device, dtype=torch.float64),
                reduction="sum",
            ).to(dtype=torch.long).cpu()

        if total_samples <= 0 or total_elements <= 0:
            raise RuntimeError("No batches were accumulated.")

        recon_mse = total_sq_error / total_elements
        mean_input = total_input_sum / total_elements
        mean_input_sq = total_input_sq_sum / total_elements
        input_variance = max(0.0, mean_input_sq - mean_input**2)
        dead_mask = fire_count == 0
        return {
            "num_samples": float(total_samples),
            "num_elements": float(total_elements),
            "recon_mse": float(recon_mse),
            "normalized_recon_mse": float(recon_mse / input_variance) if input_variance > 0.0 else float("nan"),
            "input_variance": float(input_variance),
            "loss": float(total_loss / total_samples),
            "auxk_loss": float(total_aux / total_samples),
            "avg_l0": float(total_l0 / total_samples),
            "dead_latent_count": float(dead_mask.sum().item()),
            "alive_latent_count": float((~dead_mask).sum().item()),
            "dead_latent_frac": float(dead_mask.float().mean().item()),
        }


def evaluate_reconstruction(
    model: TopKSAE,
    dataloader: DataLoader,
    *,
    device: torch.device,
    use_auxk: bool,
    auxk_alpha: float,
    dead_threshold: float,
    max_batches: int | None = None,
    accelerator: Any | None = None,
    progress_desc: str | None = None,
    show_progress: bool = False,
) -> Dict[str, float]:
    """Evaluate reconstruction quality on a held-out loader."""

    raw_model = _unwrap_sae_model(model)
    model.eval()
    accumulator = _ReconstructionAccumulator(int(getattr(raw_model, "n_latents")))
    total_batches = len(dataloader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
    batch_iterator = maybe_tqdm(
        enumerate(dataloader),
        desc=progress_desc or "sae-eval-recon",
        total=total_batches,
        enabled=show_progress,
        leave=False,
    )
    with torch.no_grad():
        for batch_index, batch in batch_iterator:
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = _coerce_batch(batch, device=device, dtype=raw_model.b_pre.dtype)
            outputs = model(inputs)
            accumulator.update(
                inputs,
                outputs,
                use_auxk=use_auxk,
                auxk_alpha=auxk_alpha,
                dead_threshold=dead_threshold,
            )
    return accumulator.summary(accelerator=accelerator, device=device)


def evaluate_sae_on_loader(
    model: TopKSAE,
    dataloader: DataLoader,
    *,
    device: torch.device,
    use_auxk: bool,
    auxk_alpha: float,
    dead_threshold: float,
    max_batches: int | None = None,
    accelerator: Any | None = None,
    progress_desc: str | None = None,
    show_progress: bool = False,
) -> Dict[str, float]:
    """Backward-compatible alias for the phase-1 evaluation entrypoint."""

    return evaluate_reconstruction(
        model,
        dataloader,
        device=device,
        use_auxk=use_auxk,
        auxk_alpha=auxk_alpha,
        dead_threshold=dead_threshold,
        max_batches=max_batches,
        accelerator=accelerator,
        progress_desc=progress_desc,
        show_progress=show_progress,
    )


def _update_value_store(
    exact_values: list[float] | None,
    sampled_values: list[float] | None,
    values: np.ndarray,
    *,
    max_exact_values: int,
    max_sample_values: int,
    rng: np.random.Generator,
) -> tuple[list[float] | None, list[float] | None]:
    if values.size == 0:
        return exact_values, sampled_values

    if exact_values is not None:
        if len(exact_values) + int(values.size) <= max_exact_values:
            exact_values.extend(values.astype(float, copy=False).tolist())
            return exact_values, sampled_values
        sampled_values = exact_values[: min(len(exact_values), max_sample_values)]
        exact_values = None

    if sampled_values is None:
        sampled_values = []

    remaining = max_sample_values - len(sampled_values)
    if remaining > 0:
        take = values[:remaining]
        sampled_values.extend(take.astype(float, copy=False).tolist())
        values = values[remaining:]

    if sampled_values and values.size > 0:
        sample_size = min(max_sample_values, int(values.size))
        sampled = rng.choice(values, size=sample_size, replace=False)
        merged = np.asarray(sampled_values + sampled.astype(float).tolist(), dtype=float)
        if merged.size > max_sample_values:
            keep = rng.choice(merged.size, size=max_sample_values, replace=False)
            merged = merged[keep]
        sampled_values = merged.astype(float, copy=False).tolist()

    return exact_values, sampled_values


def _update_top_pairs(
    heap: list[tuple[float, int, int]],
    values: torch.Tensor,
    pair_rows: torch.Tensor,
    pair_cols: torch.Tensor,
    *,
    topk_pairs: int,
) -> None:
    if values.numel() == 0 or topk_pairs <= 0:
        return

    take = min(topk_pairs, int(values.numel()))
    local_vals, local_idx = torch.topk(values, k=take)
    for score, idx in zip(local_vals.tolist(), local_idx.tolist()):
        left = int(pair_rows[idx].item())
        right = int(pair_cols[idx].item())
        item = (float(score), left, right)
        if len(heap) < topk_pairs:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)


def compute_decoder_overlap(
    model: TopKSAE,
    *,
    topk_pairs: int = 20,
    chunk_size: int = 256,
    heatmap_max_latents: int = 256,
    max_exact_values: int = 2_000_000,
    max_sample_values: int = 200_000,
    progress_desc: str | None = None,
    show_progress: bool = False,
) -> Dict[str, object]:
    """Compute cosine overlap statistics between decoder atoms.

    This SAE stores decoder atoms as row vectors in ``W_dec`` with shape
    ``[n_latents, d_in]``. The overlap summary therefore compares decoder rows.
    """

    raw_model = _unwrap_sae_model(model)
    decoder = F.normalize(raw_model.W_dec.detach().cpu().to(dtype=torch.float32), dim=1, eps=1e-8)
    n_latents = int(decoder.shape[0])
    rng = np.random.default_rng(0)
    pair_count = n_latents * (n_latents - 1) // 2
    exact_values: list[float] | None = []
    sampled_values: list[float] | None = None
    heap: list[tuple[float, int, int]] = []
    sum_values = 0.0
    total_pairs = 0
    max_score = float("-inf")
    heatmap = None

    if n_latents <= heatmap_max_latents:
        heatmap = (decoder @ decoder.T).numpy()

    row_starts = range(0, n_latents, chunk_size)
    row_iterator = maybe_tqdm(
        row_starts,
        desc=progress_desc or "sae-eval-overlap",
        total=len(row_starts),
        enabled=show_progress,
        leave=False,
    )
    for row_start in row_iterator:
        row_end = min(row_start + chunk_size, n_latents)
        left = decoder[row_start:row_end]
        for col_start in range(row_start, n_latents, chunk_size):
            col_end = min(col_start + chunk_size, n_latents)
            right = decoder[col_start:col_end]
            block = left @ right.T

            if row_start == col_start:
                tri_rows, tri_cols = torch.triu_indices(block.shape[0], block.shape[1], offset=1)
                values = block[tri_rows, tri_cols]
                pair_rows = tri_rows + row_start
                pair_cols = tri_cols + col_start
            else:
                values = block.reshape(-1)
                flat = torch.arange(values.numel(), dtype=torch.long)
                pair_rows = flat // block.shape[1] + row_start
                pair_cols = flat % block.shape[1] + col_start

            if values.numel() == 0:
                continue

            values_np = values.numpy()
            sum_values += float(values_np.sum())
            total_pairs += int(values_np.size)
            max_score = max(max_score, float(values_np.max()))
            exact_values, sampled_values = _update_value_store(
                exact_values,
                sampled_values,
                values_np,
                max_exact_values=max_exact_values,
                max_sample_values=max_sample_values,
                rng=rng,
            )
            _update_top_pairs(heap, values, pair_rows, pair_cols, topk_pairs=topk_pairs)

    if total_pairs != pair_count:
        logger.warning(
            "decoder overlap processed %d pairs but expected %d pairs",
            total_pairs,
            pair_count,
        )

    if exact_values is not None:
        exact_array = np.asarray(exact_values, dtype=float)
        median_value = float(np.median(exact_array)) if exact_array.size > 0 else float("nan")
        approximate = False
    else:
        sampled_array = np.asarray(sampled_values or [], dtype=float)
        median_value = float(np.median(sampled_array)) if sampled_array.size > 0 else float("nan")
        approximate = True

    top_pairs = sorted(heap, key=lambda item: item[0], reverse=True)
    return {
        "pair_count": float(total_pairs),
        "mean": float(sum_values / total_pairs) if total_pairs > 0 else float("nan"),
        "median": median_value,
        "max": float(max_score) if total_pairs > 0 else float("nan"),
        "approximate_median": approximate,
        "top_overlapping_pairs": format_top_pairs(
            [(left, right, score) for score, left, right in top_pairs],
            limit=topk_pairs,
        ),
        "heatmap": None if heatmap is None else heatmap.tolist(),
    }


def _summarize_pair_matrix(
    matrix: torch.Tensor,
    *,
    topk_pairs: int,
    include_heatmap: bool,
    pair_label: str,
) -> Dict[str, object]:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"pair matrix must be square, got {tuple(matrix.shape)}.")

    n_latents = int(matrix.shape[0])
    tri_rows, tri_cols = torch.triu_indices(n_latents, n_latents, offset=1)
    values = matrix[tri_rows, tri_cols]
    values_np = values.numpy()

    top_pairs_heap: list[tuple[float, int, int]] = []
    _update_top_pairs(top_pairs_heap, values, tri_rows, tri_cols, topk_pairs=topk_pairs)
    top_pairs = sorted(top_pairs_heap, key=lambda item: item[0], reverse=True)
    stats = finite_stats(values_np.tolist())

    return {
        "pair_count": stats["count"],
        "mean": stats["mean"],
        "median": stats["median"],
        "max": stats["max"],
        pair_label: format_top_pairs(
            [(left, right, score) for score, left, right in top_pairs],
            limit=topk_pairs,
        ),
        "heatmap": matrix.numpy().tolist() if include_heatmap else None,
    }


def compute_coactivation_from_latents(
    latents: torch.Tensor,
    *,
    threshold: float = 1e-8,
    topk_pairs: int = 20,
    heatmap_max_latents: int = 256,
) -> Dict[str, object]:
    """Compute latent coactivation frequencies from a dense latent matrix."""

    if latents.ndim != 2:
        raise ValueError(f"latents must have shape [num_samples, n_latents], got {tuple(latents.shape)}.")
    if int(latents.shape[0]) <= 0:
        raise RuntimeError("Cannot compute coactivation on an empty latent matrix.")

    active = (latents.abs() > threshold).to(dtype=torch.float32)
    num_samples = int(active.shape[0])
    pair_counts = active.T @ active
    freq = pair_counts / float(num_samples)
    summary = _summarize_pair_matrix(
        freq.cpu(),
        topk_pairs=topk_pairs,
        include_heatmap=int(latents.shape[1]) <= heatmap_max_latents,
        pair_label="top_coactivated_pairs",
    )
    summary.update(
        {
            "num_samples": float(num_samples),
            "activation_threshold": float(threshold),
        }
    )
    return summary


def compute_coactivation(
    model: TopKSAE,
    dataloader: DataLoader,
    *,
    device: torch.device,
    threshold: float = 1e-8,
    max_batches: int | None = None,
    topk_pairs: int = 20,
    heatmap_max_latents: int = 256,
    full_matrix_latent_limit: int = 1024,
    progress_desc: str | None = None,
    show_progress: bool = False,
) -> Dict[str, object]:
    """Compute coactivation statistics on a held-out loader."""

    raw_model = _unwrap_sae_model(model)
    model.eval()
    total_batches = len(dataloader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
    if int(getattr(raw_model, "n_latents")) <= full_matrix_latent_limit:
        latent_batches = []
        batch_iterator = maybe_tqdm(
            enumerate(dataloader),
            desc=progress_desc or "sae-eval-coact",
            total=total_batches,
            enabled=show_progress,
            leave=False,
        )
        with torch.no_grad():
            for batch_index, batch in batch_iterator:
                if max_batches is not None and batch_index >= max_batches:
                    break
                inputs = _coerce_batch(batch, device=device, dtype=raw_model.b_pre.dtype)
                latent_batches.append(model(inputs)["z"].detach().cpu())
        if not latent_batches:
            raise RuntimeError("Evaluation loader produced zero batches.")
        latents = torch.cat(latent_batches, dim=0)
        return compute_coactivation_from_latents(
            latents,
            threshold=threshold,
            topk_pairs=topk_pairs,
            heatmap_max_latents=heatmap_max_latents,
        )

    activation_counts = torch.zeros(int(getattr(raw_model, "n_latents")), dtype=torch.long)
    count_iterator = maybe_tqdm(
        enumerate(dataloader),
        desc=f"{progress_desc or 'sae-eval-coact'} count",
        total=total_batches,
        enabled=show_progress,
        leave=False,
    )
    with torch.no_grad():
        for batch_index, batch in count_iterator:
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = _coerce_batch(batch, device=device, dtype=raw_model.b_pre.dtype)
            outputs = model(inputs)
            activation_counts += (outputs["z"].detach().cpu().abs() > threshold).sum(dim=0)

    sample_latents = min(full_matrix_latent_limit, int(getattr(raw_model, "n_latents")))
    subset = torch.topk(activation_counts, k=sample_latents).indices
    subset_batches = []
    subset_iterator = maybe_tqdm(
        enumerate(dataloader),
        desc=f"{progress_desc or 'sae-eval-coact'} subset",
        total=total_batches,
        enabled=show_progress,
        leave=False,
    )
    with torch.no_grad():
        for batch_index, batch in subset_iterator:
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = _coerce_batch(batch, device=device, dtype=raw_model.b_pre.dtype)
            outputs = model(inputs)
            subset_batches.append(outputs["z"].detach().cpu()[:, subset])
    if not subset_batches:
        raise RuntimeError("Evaluation loader produced zero batches.")

    summary = compute_coactivation_from_latents(
        torch.cat(subset_batches, dim=0),
        threshold=threshold,
        topk_pairs=topk_pairs,
        heatmap_max_latents=min(heatmap_max_latents, sample_latents),
    )
    summary.update(
        {
            "approximate": True,
            "sampled_latent_count": float(sample_latents),
            "sampled_latent_indices": subset.tolist(),
        }
    )
    return summary


def summarize_sae_eval(
    reconstruction: Mapping[str, object],
    decoder_overlap: Mapping[str, object],
    coactivation: Mapping[str, object],
    *,
    context: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    """Combine scalar SAE metrics into one summary payload."""

    summary: Dict[str, object] = {}
    if context is not None:
        summary.update(dict(context))
    summary.update(
        {
            "recon_mse": reconstruction.get("recon_mse"),
            "normalized_recon_mse": reconstruction.get("normalized_recon_mse"),
            "avg_l0": reconstruction.get("avg_l0"),
            "dead_latent_frac": reconstruction.get("dead_latent_frac"),
            "decoder_overlap_mean": decoder_overlap.get("mean"),
            "decoder_overlap_median": decoder_overlap.get("median"),
            "decoder_overlap_max": decoder_overlap.get("max"),
            "coactivation_mean": coactivation.get("mean"),
            "coactivation_median": coactivation.get("median"),
            "coactivation_max": coactivation.get("max"),
        }
    )

    metrics_rows = [
        {"metric": key, "value": safe_float(value)}
        for key, value in summary.items()
        if isinstance(value, (int, float, np.floating)) and math.isfinite(float(value))
    ]
    return {
        "summary": summary,
        "metrics_rows": metrics_rows,
        "dataframe": maybe_make_dataframe(metrics_rows),
    }


class SAEEvaluator:
    """Reusable evaluator for saved SAE checkpoints."""

    def __init__(self, model: TopKSAE, sae_config: SAEConfig, *, device: torch.device) -> None:
        self.model = model
        self.sae_config = sae_config
        self.device = device

    def evaluate_loader(
        self,
        dataloader: DataLoader,
        *,
        dead_threshold: float = 1e-8,
        max_batches: int | None = None,
        accelerator: Any | None = None,
        progress_desc: str | None = None,
        show_progress: bool = False,
    ) -> Dict[str, float]:
        return evaluate_reconstruction(
            self.model,
            dataloader,
            device=self.device,
            use_auxk=self.sae_config.use_auxk,
            auxk_alpha=self.sae_config.auxk_alpha,
            dead_threshold=dead_threshold,
            max_batches=max_batches,
            accelerator=accelerator,
            progress_desc=progress_desc,
            show_progress=show_progress,
        )

    def evaluate_activation_dir(
        self,
        activation_dir: str | Path,
        *,
        batch_size: int,
        preprocessing: str,
        num_workers: int = 0,
        max_batches: int | None = None,
        dead_threshold: float = 1e-8,
        accelerator: Any | None = None,
        show_progress: bool = False,
    ) -> Dict[str, float]:
        with stage_progress("load activation shards", enabled=show_progress):
            dataset = ActivationShardDataset(activation_dir, preprocessing=preprocessing)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
        metrics = self.evaluate_loader(
            dataloader,
            dead_threshold=dead_threshold,
            max_batches=max_batches,
            accelerator=accelerator,
            progress_desc="sae-eval-recon",
            show_progress=show_progress,
        )
        metrics.update(
            {
                "activation_dir": str(Path(activation_dir).expanduser().resolve()),
                "preprocessing": preprocessing,
                "d_in": float(dataset.d_in),
            }
        )
        return metrics

    def summarize_activation_dir(
        self,
        activation_dir: str | Path,
        *,
        batch_size: int,
        preprocessing: str,
        num_workers: int = 0,
        max_batches: int | None = None,
        dead_threshold: float = 1e-8,
        topk_pairs: int = 20,
        accelerator: Any | None = None,
        show_progress: bool = False,
    ) -> Dict[str, object]:
        with stage_progress("load activation shards", enabled=show_progress):
            dataset = ActivationShardDataset(activation_dir, preprocessing=preprocessing)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
        reconstruction = self.evaluate_loader(
            dataloader,
            dead_threshold=dead_threshold,
            max_batches=max_batches,
            accelerator=accelerator,
            progress_desc="sae-eval-recon",
            show_progress=show_progress,
        )
        decoder_overlap = compute_decoder_overlap(
            self.model,
            topk_pairs=topk_pairs,
            progress_desc="sae-eval-overlap",
            show_progress=show_progress,
        )
        coactivation = compute_coactivation(
            self.model,
            dataloader,
            device=self.device,
            threshold=dead_threshold,
            max_batches=max_batches,
            topk_pairs=topk_pairs,
            progress_desc="sae-eval-coact",
            show_progress=show_progress,
        )
        raw_model = _unwrap_sae_model(self.model)
        combined = summarize_sae_eval(
            reconstruction,
            decoder_overlap,
            coactivation,
            context={
                "activation_dir": str(Path(activation_dir).expanduser().resolve()),
                "preprocessing": preprocessing,
                "d_in": dataset.d_in,
                "n_latents": int(getattr(raw_model, "n_latents")),
                "k": int(getattr(raw_model.config, "k")),
            },
        )
        return {
            "reconstruction": reconstruction,
            "decoder_overlap": decoder_overlap,
            "coactivation": coactivation,
            **combined,
        }


def save_evaluation_results(
    metrics: Mapping[str, object],
    out_dir: str | Path,
    filename: str = "metrics.json",
) -> Path:
    """Persist evaluation metrics as JSON."""

    output_dir = ensure_dir(out_dir)
    return write_json(dict(metrics), output_dir / filename)


def save_sae_eval_outputs(
    summary_payload: Mapping[str, object],
    out_dir: str | Path,
    *,
    summary_filename: str = "eval_summary.json",
    metrics_filename: str = "eval_metrics.csv",
) -> Dict[str, Path]:
    """Persist a full SAE evaluation payload."""

    output_dir = ensure_dir(out_dir)
    summary_json_payload = {key: value for key, value in summary_payload.items() if key != "dataframe"}
    summary_path = write_json(summary_json_payload, output_dir / summary_filename)
    metrics_rows = summary_payload.get("metrics_rows", [])
    metrics_path = output_dir / metrics_filename
    if isinstance(metrics_rows, Sequence):
        write_csv_rows(metrics_path, ["metric", "value"], metrics_rows)  # type: ignore[arg-type]
    else:
        write_csv_rows(metrics_path, ["metric", "value"], [])
    return {"summary": summary_path, "metrics": metrics_path}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for shard-based SAE evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate a saved SAE checkpoint on activation shards.")
    parser.add_argument("--sae-checkpoint", type=str, required=True, help="Path to best.pt / last.pt.")
    parser.add_argument("--activation-dir", type=str, required=True, help="Path to activation shard directory.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for evaluation.")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto.")
    parser.add_argument("--out-dir", type=str, default="", help="Optional custom output directory.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader worker count.")
    parser.add_argument("--preprocessing", type=str, default=None, help="Optional preprocessing override.")
    parser.add_argument("--input-centering", action="store_true", help="Apply mean-center preprocessing.")
    parser.add_argument("--input-norm", action="store_true", help="Apply unit-norm preprocessing.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional batch cap for evaluation.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for shard-based SAE evaluation."""

    configure_logging()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        device = resolve_device(args.device)
        with stage_progress("load SAE checkpoint", enabled=True):
            model, sae_config, payload = load_sae_checkpoint(args.sae_checkpoint, device=device)

        inherited_preprocessing = None
        train_cfg_payload = payload.get("train_config")
        if isinstance(train_cfg_payload, Mapping):
            inherited_preprocessing = train_cfg_payload.get("preprocessing")

        preprocessing = resolve_preprocessing_mode(
            args.preprocessing or inherited_preprocessing or "none",
            input_centering=args.input_centering,
            input_norm=args.input_norm,
        )

        eval_config = SAEEvalConfig(
            sae_checkpoint_path=args.sae_checkpoint,
            activation_dir=args.activation_dir,
            batch_size=args.batch_size,
            device=args.device,
            num_workers=args.num_workers,
            preprocessing=preprocessing,
            max_batches=args.max_batches,
            out_dir=args.out_dir,
        )

        evaluator = SAEEvaluator(model, sae_config, device=device)
        summary_payload = evaluator.summarize_activation_dir(
            eval_config.activation_dir,
            batch_size=eval_config.batch_size,
            preprocessing=preprocessing,
            num_workers=eval_config.num_workers,
            max_batches=eval_config.max_batches,
            show_progress=True,
        )

        activation_meta = read_json(Path(eval_config.activation_dir).expanduser().resolve() / "meta.json")
        source_meta = payload.get("activation_metadata", {})
        model_type = str(source_meta.get("model_type", activation_meta.get("model_type", "unknown")))
        checkpoint_step = source_meta.get("checkpoint_step", activation_meta.get("checkpoint_step"))
        layer_idx = int(source_meta.get("layer_idx", activation_meta.get("layer_idx", 0)))
        site = str(source_meta.get("site", activation_meta.get("site", "input")))
        out_dir = (
            Path(eval_config.out_dir).expanduser().resolve()
            if eval_config.out_dir
            else default_sae_eval_dir(model_type, checkpoint_step, layer_idx, site)
        )
        saved = save_sae_eval_outputs(summary_payload, out_dir)
    except Exception as error:
        logger.error("%s", error)
        return 1

    logger.info("saved SAE evaluation metrics to %s", saved["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
