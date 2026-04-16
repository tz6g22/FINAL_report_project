#!/usr/bin/env python3
"""Generate SAE analysis figures and comparison tables."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stream_analysis.sae.analysis import build_comparison_rows, save_comparison_rows
from stream_analysis.sae.utils import configure_logging, read_json, sanitize_component
from stream_analysis.sae.visualize import (
    plot_coactivation_heatmap,
    plot_dead_latents_vs_k,
    plot_decoder_overlap_heatmap,
    plot_layerwise_selective_feature_counts,
    plot_model_comparison_by_layer,
    plot_recon_vs_k,
)

logger = logging.getLogger("scripts.run_sae_visualize")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render SAE analysis comparison plots.")
    parser.add_argument(
        "--analysis-dir",
        type=str,
        nargs="+",
        required=True,
        help="One or more SAE analysis directories, or a root directory containing them.",
    )
    parser.add_argument("--out-dir", type=str, required=True, help="Directory for rendered figures and tables.")
    return parser


def _group_rows(
    rows: Sequence[Mapping[str, object]],
    key_names: Sequence[str],
) -> Dict[Tuple[object, ...], List[Mapping[str, object]]]:
    grouped: Dict[Tuple[object, ...], List[Mapping[str, object]]] = {}
    for row in rows:
        key = tuple(row.get(name) for name in key_names)
        grouped.setdefault(key, []).append(row)
    return grouped


def _finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _group_slug(prefix: str, values: Iterable[object]) -> str:
    parts = [sanitize_component(value) for value in values]
    return prefix if not parts else f"{prefix}_{'_'.join(parts)}"


def _save_run_heatmaps(rows: Sequence[Mapping[str, object]], figures_dir: Path) -> None:
    for row in rows:
        analysis_dir = Path(str(row["analysis_dir"]))
        payload = read_json(analysis_dir / "eval_summary.json")
        if not isinstance(payload, Mapping):
            continue

        model_type = sanitize_component(row.get("model_type", "unknown"))
        checkpoint_step = sanitize_component(row.get("checkpoint_step", "unknown"))
        layer_idx = sanitize_component(row.get("layer_idx", "0"))
        site = sanitize_component(row.get("site", "input"))
        stem = f"{model_type}_step_{checkpoint_step}_layer_{layer_idx}_{site}"

        decoder_overlap = payload.get("decoder_overlap", {})
        if isinstance(decoder_overlap, Mapping) and decoder_overlap.get("heatmap") is not None:
            plot_decoder_overlap_heatmap(
                decoder_overlap["heatmap"],
                figures_dir / f"decoder_overlap_heatmap_{stem}.png",
            )

        coactivation = payload.get("coactivation", {})
        if isinstance(coactivation, Mapping) and coactivation.get("heatmap") is not None:
            plot_coactivation_heatmap(
                coactivation["heatmap"],
                figures_dir / f"coactivation_heatmap_{stem}.png",
            )


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        out_dir = Path(args.out_dir).expanduser().resolve()
        figures_dir = out_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        rows = build_comparison_rows(args.analysis_dir)
        if not rows:
            raise FileNotFoundError("No eval_summary.json files were found under the requested analysis directories.")

        save_comparison_rows(rows, out_dir)
        _save_run_heatmaps(rows, figures_dir)

        k_groups = _group_rows(rows, ["model_type", "checkpoint_step", "layer_idx", "site"])
        for key, group in sorted(k_groups.items()):
            valid_group = [row for row in group if _finite_number(row.get("k")) and _finite_number(row.get("normalized_recon_mse"))]
            unique_k = sorted({float(row["k"]) for row in valid_group})
            if len(unique_k) < 2:
                continue
            slug = _group_slug("recon_vs_k", key)
            plot_recon_vs_k(valid_group, figures_dir / f"{slug}.png")
            slug = _group_slug("dead_latents_vs_k", key)
            plot_dead_latents_vs_k(valid_group, figures_dir / f"{slug}.png")

        layer_groups = _group_rows(rows, ["model_type", "checkpoint_step", "site"])
        for key, group in sorted(layer_groups.items()):
            valid_group = [
                row
                for row in group
                if _finite_number(row.get("mem_selective_feature_count")) and _finite_number(row.get("nonmem_selective_feature_count"))
            ]
            unique_layers = sorted({int(row["layer_idx"]) for row in valid_group})
            if len(unique_layers) < 2:
                continue
            slug = _group_slug("layerwise_selective_feature_counts", key)
            plot_layerwise_selective_feature_counts(valid_group, figures_dir / f"{slug}.png")

        model_groups = _group_rows(rows, ["checkpoint_step", "site"])
        for key, group in sorted(model_groups.items()):
            valid_group = [row for row in group if _finite_number(row.get("normalized_recon_mse"))]
            unique_models = sorted({str(row["model_type"]) for row in valid_group})
            unique_layers = sorted({int(row["layer_idx"]) for row in valid_group})
            if len(unique_models) < 2 or len(unique_layers) < 1:
                continue
            slug = _group_slug("model_comparison_by_layer", key)
            plot_model_comparison_by_layer(valid_group, figures_dir / f"{slug}.png")
    except Exception as error:
        logger.error("%s", error)
        return 1

    logger.info("saved SAE comparison outputs to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
