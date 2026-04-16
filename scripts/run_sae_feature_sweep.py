#!/usr/bin/env python3
"""Run a single-layer SAE feature sweep."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stream_analysis.sae.experiment import parse_feature_ids, run_sae_feature_sweep
from stream_analysis.sae.utils import configure_logging

logger = logging.getLogger("scripts.run_sae_feature_sweep")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a top-feature SAE intervention sweep.")
    parser.add_argument("--model-type", type=str, required=True, help="Model type, e.g. standard or attnres.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Source model checkpoint path.")
    parser.add_argument("--layer", type=int, required=True, help="Transformer layer index.")
    parser.add_argument("--site", type=str, default="input", help="Intervention site. Default: input.")
    parser.add_argument("--sae-checkpoint", type=str, required=True, help="Path to the trained SAE checkpoint.")
    parser.add_argument("--selection", type=str, required=True, choices=["top_mem", "top_nonmem", "manual"], help="Feature selection rule.")
    parser.add_argument("--topn", type=int, default=10, help="Number of features to sweep.")
    parser.add_argument("--dataset-split", type=str, default="analysis", help="Dataset split label for metadata.")
    parser.add_argument("--labels-source", type=str, required=True, help="Analysis-set style labels artifact.")
    parser.add_argument("--out-dir", type=str, required=True, help="Sweep output directory.")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto.")
    parser.add_argument("--feature-ids", type=str, default="", help="Comma-separated feature ids when selection=manual.")
    parser.add_argument("--mode", type=str, default="zero", choices=["zero", "keep_only", "scale"], help="Intervention mode for the sweep.")
    parser.add_argument("--scale-factor", type=float, default=0.0, help="Used when mode=scale.")
    parser.add_argument("--activation-dir", type=str, default="", help="Optional activation shard directory override.")
    parser.add_argument("--drop-error", action="store_true", help="Do not preserve SAE reconstruction error during rebuild.")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = run_sae_feature_sweep(
            model_type=args.model_type,
            checkpoint_path=args.checkpoint,
            sae_checkpoint_path=args.sae_checkpoint,
            labels_source=args.labels_source,
            layer_idx=args.layer,
            site=args.site,
            selection=args.selection,
            topn=args.topn,
            dataset_split=args.dataset_split,
            batch_size=args.batch_size,
            device=args.device,
            out_dir=args.out_dir,
            activation_dir=args.activation_dir or None,
            manual_feature_ids=parse_feature_ids(args.feature_ids),
            mode=args.mode,
            scale_factor=args.scale_factor,
            preserve_error=not args.drop_error,
        )
    except Exception as error:
        logger.error("%s", error)
        return 1

    logger.info("saved SAE feature sweep outputs to %s", result["saved_paths"].get("summary", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
