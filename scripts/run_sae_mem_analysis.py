#!/usr/bin/env python3
"""Run SAE mem/nonmem analysis on a held-out analysis set."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stream_analysis.sae.analysis import run_sae_mem_analysis
from stream_analysis.sae.utils import configure_logging, read_json
from stream_analysis.sae.visualize import plot_coactivation_heatmap, plot_decoder_overlap_heatmap

logger = logging.getLogger("scripts.run_sae_mem_analysis")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run held-out mem/nonmem analysis for a trained SAE.")
    parser.add_argument("--sae-checkpoint", type=str, required=True, help="Path to a trained SAE checkpoint.")
    parser.add_argument("--activation-dir", type=str, required=True, help="Activation shard directory used for context and eval.")
    parser.add_argument("--labels-source", type=str, required=True, help="Analysis-set style artifact with mem/nonmem labels.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for held-out extraction and evaluation.")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto.")
    parser.add_argument("--out-dir", type=str, default="", help="Optional custom analysis output directory.")
    parser.add_argument("--topk-examples", type=int, default=10, help="Top activating examples to keep per feature.")
    parser.add_argument("--dead-threshold", type=float, default=1e-8, help="Activity threshold used for L0/dead-latent stats.")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = run_sae_mem_analysis(
            args.sae_checkpoint,
            args.activation_dir,
            args.labels_source,
            batch_size=args.batch_size,
            device=args.device,
            out_dir=args.out_dir or None,
            topk_examples=args.topk_examples,
            dead_threshold=args.dead_threshold,
        )

        out_dir = Path(result["out_dir"])
        payload = read_json(out_dir / "eval_summary.json")
        if not isinstance(payload, dict):
            raise TypeError("eval_summary.json must contain a JSON object.")

        figures_dir = out_dir / "figures"
        decoder_overlap = payload.get("decoder_overlap", {})
        if isinstance(decoder_overlap, dict) and decoder_overlap.get("heatmap") is not None:
            plot_decoder_overlap_heatmap(
                decoder_overlap["heatmap"],
                figures_dir / "decoder_overlap_heatmap.png",
            )
        coactivation = payload.get("coactivation", {})
        if isinstance(coactivation, dict) and coactivation.get("heatmap") is not None:
            plot_coactivation_heatmap(
                coactivation["heatmap"],
                figures_dir / "coactivation_heatmap.png",
            )
    except Exception as error:
        logger.error("%s", error)
        return 1

    logger.info("saved SAE mem analysis outputs to %s", result["out_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
