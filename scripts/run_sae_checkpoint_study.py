#!/usr/bin/env python3
"""Run a checkpoint-wise SAE intervention study."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stream_analysis.sae.experiment import run_sae_checkpoint_study
from stream_analysis.sae.utils import configure_logging

logger = logging.getLogger("scripts.run_sae_checkpoint_study")


def _parse_int_list(value: str) -> list[int]:
    items = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not items:
        raise ValueError("Expected a non-empty comma-separated integer list.")
    return [int(item) for item in items]


def _parse_str_list(value: str) -> list[str]:
    items = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not items:
        raise ValueError("Expected a non-empty comma-separated string list.")
    return items


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a checkpoint-wise SAE intervention study.")
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        help="One model type or a comma-separated list, e.g. standard or standard,attnres.",
    )
    parser.add_argument("--layers", type=str, required=True, help="Comma-separated layer indices.")
    parser.add_argument("--sites", type=str, required=True, help="Comma-separated intervention sites.")
    parser.add_argument("--sae-root", type=str, required=True, help="Root directory containing trained SAE checkpoints.")
    parser.add_argument("--checkpoints", type=str, required=True, help="Comma-separated checkpoint step identifiers.")
    parser.add_argument("--labels-source", type=str, required=True, help="Analysis-set style labels artifact.")
    parser.add_argument("--out-dir", type=str, required=True, help="Study output directory.")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto.")
    parser.add_argument("--selection", type=str, default="top_mem", choices=["top_mem", "top_nonmem"], help="Feature-selection rule per checkpoint.")
    parser.add_argument("--topn", type=int, default=1, help="Number of features to evaluate per checkpoint/layer/site.")
    parser.add_argument("--mode", type=str, default="zero", choices=["zero", "keep_only", "scale"], help="Intervention mode.")
    parser.add_argument("--scale-factor", type=float, default=0.0, help="Used when mode=scale.")
    parser.add_argument("--drop-error", action="store_true", help="Do not preserve SAE reconstruction error during rebuild.")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = run_sae_checkpoint_study(
            model_type=_parse_str_list(args.model_type),
            layers=_parse_int_list(args.layers),
            sites=_parse_str_list(args.sites),
            sae_root=args.sae_root,
            checkpoints=_parse_str_list(args.checkpoints),
            labels_source=args.labels_source,
            out_dir=args.out_dir,
            batch_size=args.batch_size,
            device=args.device,
            selection=args.selection,
            topn=args.topn,
            mode=args.mode,
            scale_factor=args.scale_factor,
            preserve_error=not args.drop_error,
        )
    except Exception as error:
        logger.error("%s", error)
        return 1

    logger.info("saved SAE checkpoint study outputs to %s", result["saved_paths"].get("summary", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
