#!/usr/bin/env python3
"""Training loop and CLI for TopK SAE models."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import torch
from torch.utils.data import DataLoader
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .config import SAEConfig, SAETrainConfig
from .data import ActivationShardDataset, build_activation_dataloaders
from .eval import evaluate_sae_on_loader
from .losses import compute_loss_dict
from .model import TopKSAE
from .utils import (
    append_csv_row,
    configure_logging,
    default_sae_checkpoint_dir,
    ensure_dir,
    read_json,
    resolve_device,
    seed_everything,
    write_json,
)

logger = logging.getLogger("sae.train")


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


class SAETrainer:
    """Trainer for single-layer, single-site SAE fitting."""

    CSV_FIELDS = [
        "step",
        "train_loss",
        "train_recon_mse",
        "train_auxk_loss",
        "train_avg_l0",
        "train_dead_latent_frac",
        "val_loss",
        "val_recon_mse",
        "val_auxk_loss",
        "val_avg_l0",
        "val_dead_latent_frac",
        "learning_rate",
    ]

    def __init__(
        self,
        model: TopKSAE,
        *,
        sae_config: SAEConfig,
        train_config: SAETrainConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str | Path,
        device: torch.device,
        activation_metadata: Mapping[str, object] | None = None,
        activation_dir: str | Path | None = None,
    ) -> None:
        self.model = model
        self.sae_config = sae_config
        self.train_config = train_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = ensure_dir(output_dir)
        self.device = device
        self.activation_metadata = {} if activation_metadata is None else dict(activation_metadata)
        self.activation_dir = None if activation_dir is None else str(Path(activation_dir).expanduser().resolve())
        self.optimizer = self._build_optimizer()
        self.metrics_path = self.output_dir / train_config.metrics_filename
        self.best_val_recon_mse = math.inf
        self.last_val_metrics: Dict[str, float] | None = None

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = self.model.parameters()
        kwargs = {
            "lr": self.train_config.lr,
            "betas": (self.train_config.beta1, self.train_config.beta2),
            "weight_decay": self.train_config.weight_decay,
        }
        if self.train_config.optimizer == "adam":
            return torch.optim.Adam(params, **kwargs)
        return torch.optim.AdamW(params, **kwargs)

    def _run_train_step(self, batch: object) -> Dict[str, float]:
        inputs = _coerce_batch(batch, device=self.device, dtype=self.model.b_pre.dtype)
        self.model.train()
        outputs = self.model(inputs)
        losses = compute_loss_dict(
            inputs,
            outputs,
            use_auxk=self.sae_config.use_auxk,
            auxk_alpha=self.sae_config.auxk_alpha,
            dead_threshold=self.train_config.dead_threshold,
        )

        self.optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        if self.train_config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.grad_clip)
        self.optimizer.step()
        if self.sae_config.normalize_decoder:
            self.model.normalize_decoder_weights()

        return {
            "loss": float(losses["loss"].item()),
            "recon_mse": float(losses["recon_mse"].item()),
            "auxk_loss": float(losses["auxk_loss"].item()),
            "avg_l0": float(losses["avg_l0"].item()),
            "dead_latent_frac": float(losses["dead_latent_frac"]),
        }

    def _evaluate(self) -> Dict[str, float]:
        metrics = evaluate_sae_on_loader(
            self.model,
            self.val_loader,
            device=self.device,
            use_auxk=self.sae_config.use_auxk,
            auxk_alpha=self.sae_config.auxk_alpha,
            dead_threshold=self.train_config.dead_threshold,
            max_batches=self.train_config.max_val_batches,
        )
        self.last_val_metrics = metrics
        return metrics

    def _save_checkpoint(self, name: str, *, step: int) -> Path:
        path = self.output_dir / name
        payload = {
            "step": step,
            "sae_config": self.sae_config.to_dict(),
            "train_config": self.train_config.to_dict(),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_recon_mse": self.best_val_recon_mse,
            "activation_metadata": self.activation_metadata,
            "activation_dir": self.activation_dir,
        }
        torch.save(payload, path)
        return path

    def _write_config(self) -> Path:
        payload = {
            "sae_config": self.sae_config.to_dict(),
            "train_config": self.train_config.to_dict(),
            "activation_metadata": self.activation_metadata,
            "activation_dir": self.activation_dir,
        }
        return write_json(payload, self.output_dir / "config.json")

    def train(self) -> Dict[str, Any]:
        """Run training end to end and return a summary payload."""

        seed_everything(self.train_config.seed)
        self._write_config()

        train_iterator = iter(self.train_loader)
        progress = tqdm(range(1, self.train_config.num_steps + 1), desc="sae-train") if tqdm else range(
            1, self.train_config.num_steps + 1
        )

        for step in progress:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_loader)
                batch = next(train_iterator)

            train_metrics = self._run_train_step(batch)

            should_eval = step % self.train_config.eval_interval == 0 or step == self.train_config.num_steps
            if should_eval:
                val_metrics = self._evaluate()
                if val_metrics["recon_mse"] < self.best_val_recon_mse:
                    self.best_val_recon_mse = val_metrics["recon_mse"]
                    best_path = self._save_checkpoint("best.pt", step=step)
                    logger.info("new best checkpoint at step=%d saved to %s", step, best_path)
            else:
                val_metrics = self.last_val_metrics or {
                    "loss": float("nan"),
                    "recon_mse": float("nan"),
                    "auxk_loss": float("nan"),
                    "avg_l0": float("nan"),
                    "dead_latent_frac": float("nan"),
                }

            row = {
                "step": step,
                "train_loss": train_metrics["loss"],
                "train_recon_mse": train_metrics["recon_mse"],
                "train_auxk_loss": train_metrics["auxk_loss"],
                "train_avg_l0": train_metrics["avg_l0"],
                "train_dead_latent_frac": train_metrics["dead_latent_frac"],
                "val_loss": float(val_metrics["loss"]),
                "val_recon_mse": float(val_metrics["recon_mse"]),
                "val_auxk_loss": float(val_metrics["auxk_loss"]),
                "val_avg_l0": float(val_metrics["avg_l0"]),
                "val_dead_latent_frac": float(val_metrics["dead_latent_frac"]),
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            }
            append_csv_row(self.metrics_path, self.CSV_FIELDS, row)

            should_ckpt = step % self.train_config.checkpoint_interval == 0 or step == self.train_config.num_steps
            if should_ckpt:
                last_path = self._save_checkpoint("last.pt", step=step)
                logger.info("saved checkpoint at step=%d to %s", step, last_path)

            if tqdm and hasattr(progress, "set_postfix"):
                progress.set_postfix(
                    train_recon=f"{train_metrics['recon_mse']:.4f}",
                    val_recon=f"{float(val_metrics['recon_mse']):.4f}" if not math.isnan(float(val_metrics["recon_mse"])) else "nan",
                )

        return {
            "output_dir": str(self.output_dir),
            "best_val_recon_mse": self.best_val_recon_mse,
            "metrics_path": str(self.metrics_path),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a TopK SAE on extracted activation shards.")
    parser.add_argument("--activation-dir", type=str, required=True, help="Activation shard directory with meta.json.")
    parser.add_argument("--n-latents", type=int, required=True, help="Number of SAE latents.")
    parser.add_argument("--k", type=int, required=True, help="Top-k sparsity level.")
    parser.add_argument("--batch-size", type=int, default=512, help="Training batch size.")
    parser.add_argument("--num-steps", type=int, default=10000, help="Number of optimization steps.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--out-dir", type=str, default="", help="Optional custom checkpoint directory.")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto.")
    parser.add_argument("--use-auxk", action="store_true", help="Enable the auxiliary inactive-tail penalty.")
    parser.add_argument("--normalize-decoder", action="store_true", help="Project decoder atoms to unit norm after each step.")
    parser.add_argument("--input-centering", action="store_true", help="Apply mean-center preprocessing.")
    parser.add_argument("--input-norm", action="store_true", help="Apply unit-norm preprocessing.")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"], help="Optimizer type.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--eval-interval", type=int, default=250, help="Validation interval in steps.")
    parser.add_argument("--log-interval", type=int, default=25, help="Logging interval in steps.")
    parser.add_argument("--checkpoint-interval", type=int, default=500, help="Checkpoint interval in steps.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Held-out validation fraction.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader worker count.")
    parser.add_argument("--auxk-alpha", type=float, default=0.0, help="AuxK loss coefficient.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for SAE training."""

    configure_logging()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        activation_dir = Path(args.activation_dir).expanduser().resolve()
        activation_meta = read_json(activation_dir / "meta.json")
        if not isinstance(activation_meta, Mapping):
            raise TypeError("Activation meta.json must contain a JSON object.")

        preprocessing = "none"
        if args.input_centering and args.input_norm:
            preprocessing = "mean-center+unit-norm"
        elif args.input_centering:
            preprocessing = "mean-center"
        elif args.input_norm:
            preprocessing = "unit-norm"

        train_config = SAETrainConfig(
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            lr=args.lr,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            checkpoint_interval=args.checkpoint_interval,
            val_fraction=args.val_fraction,
            seed=args.seed,
            num_workers=args.num_workers,
            preprocessing=preprocessing,
            input_centering=args.input_centering,
            input_norm=args.input_norm,
            device=args.device,
        )
        sae_config = SAEConfig(
            d_in=int(activation_meta["d_model"]),
            n_latents=args.n_latents,
            k=args.k,
            use_auxk=args.use_auxk,
            auxk_alpha=args.auxk_alpha,
            normalize_decoder=args.normalize_decoder,
            device=str(resolve_device(args.device)),
        )
        device = resolve_device(args.device)

        train_loader, val_loader, _, _ = build_activation_dataloaders(
            activation_dir,
            batch_size=train_config.batch_size,
            val_fraction=train_config.val_fraction,
            seed=train_config.seed,
            preprocessing=train_config.preprocessing,
            num_workers=train_config.num_workers,
        )
        model = TopKSAE(sae_config).to(device)

        model_type = str(activation_meta.get("model_type", "unknown"))
        checkpoint_step = activation_meta.get("checkpoint_step")
        layer_idx = int(activation_meta.get("layer_idx", 0))
        site = str(activation_meta.get("site", "input"))
        output_dir = (
            Path(args.out_dir).expanduser().resolve()
            if args.out_dir
            else default_sae_checkpoint_dir(model_type, checkpoint_step, layer_idx, site)
        )

        trainer = SAETrainer(
            model,
            sae_config=sae_config,
            train_config=train_config,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=output_dir,
            device=device,
            activation_metadata=activation_meta,
            activation_dir=activation_dir,
        )
        summary = trainer.train()
    except Exception as error:
        logger.error("%s", error)
        return 1

    logger.info("training finished: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
