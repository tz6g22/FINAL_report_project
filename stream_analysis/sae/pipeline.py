#!/usr/bin/env python3
"""One-command dual-model SAE pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
from accelerate import Accelerator

from .config import SAEExtractConfig, resolve_preprocessing_mode
from .eval import SAEEvaluator, load_sae_checkpoint, save_sae_eval_outputs
from .extract import extract_activation_shards
from .train import train_sae_from_activation_dir
from .utils import configure_logging, ensure_dir, stage_progress, write_json

logger = logging.getLogger("sae.pipeline")

_MODEL_ORDER = ("standard", "attnres")


def _default_checkpoint_path(model_type: str) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "toygpt2_runs" / "tinystories_dual" / model_type / f"ckpt_{model_type}_last.pt"


def _resolve_checkpoint_paths(args: argparse.Namespace) -> Dict[str, Path]:
    checkpoints = {
        "standard": Path(args.standard_checkpoint).expanduser().resolve()
        if args.standard_checkpoint
        else _default_checkpoint_path("standard"),
        "attnres": Path(args.attnres_checkpoint).expanduser().resolve()
        if args.attnres_checkpoint
        else _default_checkpoint_path("attnres"),
    }
    missing = [f"{model_type}: {path}" for model_type, path in checkpoints.items() if not path.is_file()]
    if missing:
        joined = "; ".join(missing)
        raise FileNotFoundError(f"Dual-model pipeline requires both checkpoints. Missing: {joined}")
    return checkpoints


def _model_dirs(output_root: Path, model_type: str) -> Dict[str, Path]:
    model_root = output_root / model_type
    return {
        "root": model_root,
        "train_activations": model_root / "activations" / "train",
        "val_activations": model_root / "activations" / "val",
        "sae": model_root / "sae",
        "eval": model_root / "eval",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run train-extract, val-extract, SAE training, and SAE eval for standard + attnres in one command."
    )
    parser.add_argument("--standard-checkpoint", type=str, default="", help="Optional override for the standard checkpoint.")
    parser.add_argument("--attnres-checkpoint", type=str, default="", help="Optional override for the attnres checkpoint.")
    parser.add_argument("--layer", type=int, default=0, help="Transformer layer index.")
    parser.add_argument("--site", type=str, default="input", help="Activation site. Default: input.")
    parser.add_argument("--train-max-tokens", type=int, default=8192, help="Token budget for SAE train activations.")
    parser.add_argument("--val-max-tokens", type=int, default=4096, help="Token budget for held-out eval activations.")
    parser.add_argument("--extract-batch-size", type=int, default=8, help="Batch size for activation extraction.")
    parser.add_argument("--extract-num-workers", type=int, default=0, help="Dataloader workers for extraction.")
    parser.add_argument("--n-latents", type=int, required=True, help="Number of SAE latents.")
    parser.add_argument("--k", type=int, required=True, help="Top-k sparsity level.")
    parser.add_argument("--sae-batch-size", type=int, default=512, help="SAE training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=512, help="SAE evaluation batch size.")
    parser.add_argument("--train-num-workers", type=int, default=0, help="Dataloader workers for SAE train/eval.")
    parser.add_argument("--num-steps", type=int, default=2000, help="Number of SAE optimization steps.")
    parser.add_argument("--lr", type=float, default=3e-4, help="SAE learning rate.")
    parser.add_argument("--eval-interval", type=int, default=250, help="Validation interval during SAE training.")
    parser.add_argument("--checkpoint-interval", type=int, default=500, help="Checkpoint interval during SAE training.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Held-out split fraction inside train activations.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Extraction / eval device. Default: cuda.")
    parser.add_argument("--normalize-decoder", action="store_true", help="Normalize decoder atoms after each update.")
    parser.add_argument("--input-centering", action="store_true", help="Apply mean-center preprocessing to activations.")
    parser.add_argument("--input-norm", action="store_true", help="Apply unit-norm preprocessing to activations.")
    parser.add_argument("--use-auxk", action="store_true", help="Enable AuxK loss.")
    parser.add_argument("--auxk-alpha", type=float, default=0.0, help="AuxK loss weight.")
    parser.add_argument("--out-dir", type=str, default="outputs/sae_dual_pipeline", help="Root directory for all dual-model outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting activation shard directories.")
    return parser


def run_dual_sae_pipeline(
    args: argparse.Namespace,
    *,
    accelerator: Accelerator | None = None,
) -> Dict[str, Any]:
    accelerator = Accelerator() if accelerator is None else accelerator
    output_root = ensure_dir(Path(args.out_dir).expanduser().resolve())
    checkpoints = _resolve_checkpoint_paths(args)
    preprocessing = resolve_preprocessing_mode(
        "none",
        input_centering=args.input_centering,
        input_norm=args.input_norm,
    )

    pipeline_summary: Dict[str, Any] = {
        "output_root": str(output_root),
        "layer_idx": int(args.layer),
        "site": str(args.site),
        "accelerate": {
            "num_processes": int(accelerator.num_processes),
            "device": str(accelerator.device),
        },
        "sae_hparams": {
            "n_latents": int(args.n_latents),
            "k": int(args.k),
            "sae_batch_size": int(args.sae_batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "num_steps": int(args.num_steps),
            "lr": float(args.lr),
            "normalize_decoder": bool(args.normalize_decoder),
            "preprocessing": preprocessing,
            "use_auxk": bool(args.use_auxk),
            "auxk_alpha": float(args.auxk_alpha),
        },
        "models": {},
    }

    for model_type in _MODEL_ORDER:
        checkpoint_path = checkpoints[model_type]
        dirs = _model_dirs(output_root, model_type)
        if accelerator.is_main_process:
            logger.info("[%s] extract train activations", model_type)
            extract_activation_shards(
                SAEExtractConfig(
                    model_type=model_type,
                    checkpoint_path=str(checkpoint_path),
                    layer_idx=args.layer,
                    site=args.site,
                    dataset_split="train",
                    max_tokens=args.train_max_tokens,
                    out_dir=str(dirs["train_activations"]),
                    batch_size=args.extract_batch_size,
                    device=args.device,
                    num_workers=args.extract_num_workers,
                    overwrite=args.overwrite,
                ),
                show_progress=True,
            )
            logger.info("[%s] extract val activations", model_type)
            extract_activation_shards(
                SAEExtractConfig(
                    model_type=model_type,
                    checkpoint_path=str(checkpoint_path),
                    layer_idx=args.layer,
                    site=args.site,
                    dataset_split="val",
                    max_tokens=args.val_max_tokens,
                    out_dir=str(dirs["val_activations"]),
                    batch_size=args.extract_batch_size,
                    device=args.device,
                    num_workers=args.extract_num_workers,
                    overwrite=args.overwrite,
                ),
                show_progress=True,
            )

        accelerator.wait_for_everyone()

        train_summary = train_sae_from_activation_dir(
            dirs["train_activations"],
            n_latents=args.n_latents,
            k=args.k,
            batch_size=args.sae_batch_size,
            num_steps=args.num_steps,
            lr=args.lr,
            out_dir=dirs["sae"],
            device=args.device,
            use_auxk=args.use_auxk,
            normalize_decoder=args.normalize_decoder,
            input_centering=args.input_centering,
            input_norm=args.input_norm,
            eval_interval=args.eval_interval,
            checkpoint_interval=args.checkpoint_interval,
            val_fraction=args.val_fraction,
            seed=args.seed,
            num_workers=args.train_num_workers,
            auxk_alpha=args.auxk_alpha,
            accelerator=accelerator,
            show_progress=accelerator.is_main_process,
        )
        accelerator.wait_for_everyone()

        model_summary: Dict[str, Any] = {
            "checkpoint_path": str(checkpoint_path),
            "train_activation_dir": str(dirs["train_activations"]),
            "val_activation_dir": str(dirs["val_activations"]),
            "sae_dir": str(dirs["sae"]),
            "sae_best_checkpoint": str(dirs["sae"] / "best.pt"),
            "sae_last_checkpoint": str(dirs["sae"] / "last.pt"),
            "metrics_path": str(dirs["sae"] / "metrics.csv"),
            "best_val_recon_mse": float(train_summary["best_val_recon_mse"]),
        }

        if accelerator.is_main_process:
            logger.info("[%s] evaluate SAE checkpoint", model_type)
            with stage_progress(f"[{model_type}] load SAE checkpoint", enabled=True):
                sae_model, sae_config, _ = load_sae_checkpoint(dirs["sae"] / "best.pt", device=accelerator.device)
            evaluator = SAEEvaluator(sae_model, sae_config, device=accelerator.device)
            summary_payload = evaluator.summarize_activation_dir(
                dirs["val_activations"],
                batch_size=args.eval_batch_size,
                preprocessing=preprocessing,
                num_workers=args.train_num_workers,
                show_progress=True,
            )
            saved = save_sae_eval_outputs(summary_payload, dirs["eval"])
            model_summary.update(
                {
                    "eval_dir": str(dirs["eval"]),
                    "eval_summary_path": str(saved["summary"]),
                    "eval_metrics_path": str(saved["metrics"]),
                }
            )
            pipeline_summary["models"][model_type] = model_summary
            write_json(pipeline_summary, output_root / "pipeline_summary.json")

        accelerator.wait_for_everyone()
        if hasattr(accelerator, "free_memory"):
            accelerator.free_memory()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if accelerator.is_main_process:
        write_json(pipeline_summary, output_root / "pipeline_summary.json")
    accelerator.wait_for_everyone()
    return pipeline_summary


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    accelerator = Accelerator()

    if not accelerator.is_main_process:
        logging.getLogger().setLevel(logging.WARNING)

    try:
        summary = run_dual_sae_pipeline(args, accelerator=accelerator)
    except Exception as error:
        logger.error("%s", error)
        return 1

    if accelerator.is_main_process:
        logger.info("dual SAE pipeline finished: %s", summary["output_root"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
