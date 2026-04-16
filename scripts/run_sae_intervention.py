#!/usr/bin/env python3
"""Run one SAE feature intervention experiment."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stream_analysis.sae.experiment import parse_feature_ids, run_sae_intervention_experiment
from stream_analysis.sae.utils import configure_logging

logger = logging.getLogger("scripts.run_sae_intervention")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single SAE feature intervention experiment.")
    parser.add_argument("--model-type", type=str, required=True, help="Model type, e.g. standard or attnres.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Source model checkpoint path.")
    parser.add_argument("--layer", type=int, required=True, help="Transformer layer index.")
    parser.add_argument("--site", type=str, default="input", help="Intervention site. Default: input.")
    parser.add_argument("--sae-checkpoint", type=str, required=True, help="Path to the trained SAE checkpoint.")
    parser.add_argument("--mode", type=str, required=True, choices=["zero", "keep_only", "scale", "patch"], help="Feature intervention mode.")
    parser.add_argument("--feature-ids", type=str, required=True, help="Comma-separated feature ids.")
    parser.add_argument("--dataset-split", type=str, default="analysis", help="Dataset split label for metadata.")
    parser.add_argument("--labels-source", type=str, required=True, help="Analysis-set style labels artifact.")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto.")
    parser.add_argument("--out-dir", type=str, default="", help="Optional custom output directory.")
    parser.add_argument("--scale-factor", type=float, default=0.0, help="Used when mode=scale. Ignored otherwise.")
    parser.add_argument("--donor-labels-source", type=str, default="", help="Required when mode=patch.")
    parser.add_argument("--activation-dir", type=str, default="", help="Optional activation shard directory override.")
    parser.add_argument("--drop-error", action="store_true", help="Do not preserve SAE reconstruction error during rebuild.")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = run_sae_intervention_experiment(
            model_type=args.model_type,
            checkpoint_path=args.checkpoint,
            sae_checkpoint_path=args.sae_checkpoint,
            labels_source=args.labels_source,
            layer_idx=args.layer,
            site=args.site,
            mode=args.mode,
            feature_ids=parse_feature_ids(args.feature_ids),
            dataset_split=args.dataset_split,
            batch_size=args.batch_size,
            device=args.device,
            out_dir=args.out_dir or None,
            activation_dir=args.activation_dir or None,
            scale_factor=args.scale_factor,
            preserve_error=not args.drop_error,
            donor_labels_source=args.donor_labels_source or None,
            write_outputs=True,
        )
    except Exception as error:
        logger.error("%s", error)
        return 1

    logger.info("saved SAE intervention outputs to %s", result["saved_paths"].get("summary", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
