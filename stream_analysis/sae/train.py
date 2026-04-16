#!/usr/bin/env python3
"""Training loop and CLI for TopK SAE models."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .config import SAEConfig, SAETrainConfig
from .data import build_activation_dataloaders
from .eval import evaluate_sae_on_loader
from .losses import compute_loss_dict
from .model import TopKSAE
from .utils import (
    append_csv_row,
    configure_logging,
    default_sae_checkpoint_dir,
    ensure_dir,
    maybe_tqdm,
    read_json,
    resolve_device,
    seed_everything,
    stage_progress,
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


def _mean_reduce(
    values: Mapping[str, float],
    *,
    batch_size: int,
    device: torch.device,
    accelerator: Accelerator | None,
) -> Dict[str, float]:
    weight = float(batch_size)
    totals = torch.tensor(
        [
            values["loss"] * weight,
            values["recon_mse"] * weight,
            values["auxk_loss"] * weight,
            values["avg_l0"] * weight,
            values["dead_latent_frac"] * weight,
            weight,
        ],
        device=device,
        dtype=torch.float64,
    )
    if accelerator is not None and accelerator.num_processes > 1:
        totals = accelerator.reduce(totals, reduction="sum")
    denom = max(float(totals[-1].item()), 1.0)
    return {
        "loss": float(totals[0].item() / denom),
        "recon_mse": float(totals[1].item() / denom),
        "auxk_loss": float(totals[2].item() / denom),
        "avg_l0": float(totals[3].item() / denom),
        "dead_latent_frac": float(totals[4].item() / denom),
    }


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
        device: torch.device | None = None,
        activation_metadata: Mapping[str, object] | None = None,
        activation_dir: str | Path | None = None,
        accelerator: Accelerator | None = None,
        show_progress: bool | None = None,
    ) -> None:
        self.accelerator = accelerator
        self.model = model
        self.sae_config = sae_config
        self.train_config = train_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.activation_metadata = {} if activation_metadata is None else dict(activation_metadata)
        self.activation_dir = None if activation_dir is None else str(Path(activation_dir).expanduser().resolve())
        self.device = accelerator.device if accelerator is not None else (
            torch.device("cpu") if device is None else device
        )
        self.is_main_process = True if accelerator is None else bool(accelerator.is_main_process)
        self.show_progress = self.is_main_process if show_progress is None else bool(show_progress)
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

    def _unwrap_model(self) -> TopKSAE:
        if self.accelerator is None:
            return self.model
        return self.accelerator.unwrap_model(self.model)

    def _maybe_wait(self) -> None:
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    def _save_checkpoint(self, name: str, *, step: int) -> Path | None:
        if not self.is_main_process:
            return None

        path = self.output_dir / name
        payload = {
            "step": step,
            "sae_config": self.sae_config.to_dict(),
            "train_config": self.train_config.to_dict(),
            "model_state": self._unwrap_model().state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_recon_mse": self.best_val_recon_mse,
            "activation_metadata": self.activation_metadata,
            "activation_dir": self.activation_dir,
        }
        torch.save(payload, path)
        return path

    def _write_config(self) -> Path | None:
        if not self.is_main_process:
            return None

        payload = {
            "sae_config": self.sae_config.to_dict(),
            "train_config": self.train_config.to_dict(),
            "activation_metadata": self.activation_metadata,
            "activation_dir": self.activation_dir,
        }
        return write_json(payload, self.output_dir / "config.json")

    def _run_train_step(self, batch: object) -> Dict[str, float]:
        inputs = _coerce_batch(batch, device=self.device, dtype=self._unwrap_model().b_pre.dtype)
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
        if self.accelerator is None:
            losses["loss"].backward()
            if self.train_config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.grad_clip)
        else:
            self.accelerator.backward(losses["loss"])
            if self.train_config.grad_clip is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.train_config.grad_clip)
        self.optimizer.step()
        if self.sae_config.normalize_decoder:
            self._unwrap_model().normalize_decoder_weights()

        local_metrics = {
            "loss": float(losses["loss"].item()),
            "recon_mse": float(losses["recon_mse"].item()),
            "auxk_loss": float(losses["auxk_loss"].item()),
            "avg_l0": float(losses["avg_l0"].item()),
            "dead_latent_frac": float(losses["dead_latent_frac"]),
        }
        return _mean_reduce(
            local_metrics,
            batch_size=int(inputs.shape[0]),
            device=self.device,
            accelerator=self.accelerator,
        )

    def _evaluate(self) -> Dict[str, float]:
        metrics = evaluate_sae_on_loader(
            self.model,
            self.val_loader,
            device=self.device,
            use_auxk=self.sae_config.use_auxk,
            auxk_alpha=self.sae_config.auxk_alpha,
            dead_threshold=self.train_config.dead_threshold,
            max_batches=self.train_config.max_val_batches,
            accelerator=self.accelerator,
            progress_desc="sae-val",
            show_progress=self.show_progress,
        )
        self.last_val_metrics = metrics
        return metrics

    def train(self) -> Dict[str, Any]:
        """Run training end to end and return a summary payload."""

        seed_everything(self.train_config.seed)
        if self.is_main_process:
            ensure_dir(self.output_dir)
            if self.metrics_path.exists():
                self.metrics_path.unlink()
            self._write_config()
        self._maybe_wait()

        train_iterator = iter(self.train_loader)
        progress = maybe_tqdm(
            range(1, self.train_config.num_steps + 1),
            desc="sae-train",
            enabled=self.show_progress,
            leave=False,
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
                    if self.is_main_process and best_path is not None:
                        logger.info("new best checkpoint at step=%d saved to %s", step, best_path)
                self._maybe_wait()
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
            if self.is_main_process:
                append_csv_row(self.metrics_path, self.CSV_FIELDS, row)

            should_ckpt = step % self.train_config.checkpoint_interval == 0 or step == self.train_config.num_steps
            if should_ckpt:
                last_path = self._save_checkpoint("last.pt", step=step)
                if self.is_main_process and last_path is not None:
                    logger.info("saved checkpoint at step=%d to %s", step, last_path)
                self._maybe_wait()

            if hasattr(progress, "set_postfix"):
                progress.set_postfix(
                    train_recon=f"{train_metrics['recon_mse']:.4f}",
                    val_recon=(
                        f"{float(val_metrics['recon_mse']):.4f}"
                        if not math.isnan(float(val_metrics["recon_mse"]))
                        else "nan"
                    ),
                )

        return {
            "output_dir": str(self.output_dir),
            "best_val_recon_mse": self.best_val_recon_mse,
            "metrics_path": str(self.metrics_path),
        }


def train_sae_from_activation_dir(
    activation_dir: str | Path,
    *,
    n_latents: int,
    k: int,
    batch_size: int = 512,
    num_steps: int = 10_000,
    lr: float = 3e-4,
    out_dir: str | Path | None = None,
    device: str = "auto",
    use_auxk: bool = False,
    normalize_decoder: bool = False,
    input_centering: bool = False,
    input_norm: bool = False,
    optimizer: str = "adamw",
    weight_decay: float = 0.0,
    eval_interval: int = 250,
    log_interval: int = 25,
    checkpoint_interval: int = 500,
    val_fraction: float = 0.1,
    seed: int = 1234,
    num_workers: int = 0,
    auxk_alpha: float = 0.0,
    accelerator: Accelerator | None = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Train one SAE from an activation shard directory."""

    resolved_activation_dir = Path(activation_dir).expanduser().resolve()
    with stage_progress("load activation metadata", enabled=show_progress):
        activation_meta = read_json(resolved_activation_dir / "meta.json")
    if not isinstance(activation_meta, Mapping):
        raise TypeError("Activation meta.json must contain a JSON object.")

    preprocessing = "none"
    if input_centering and input_norm:
        preprocessing = "mean-center+unit-norm"
    elif input_centering:
        preprocessing = "mean-center"
    elif input_norm:
        preprocessing = "unit-norm"

    train_config = SAETrainConfig(
        batch_size=batch_size,
        num_steps=num_steps,
        lr=lr,
        optimizer=optimizer,
        weight_decay=weight_decay,
        eval_interval=eval_interval,
        log_interval=log_interval,
        checkpoint_interval=checkpoint_interval,
        val_fraction=val_fraction,
        seed=seed,
        num_workers=num_workers,
        preprocessing=preprocessing,
        input_centering=input_centering,
        input_norm=input_norm,
        device=device,
    )

    device_obj = resolve_device(device) if accelerator is None else accelerator.device
    sae_config = SAEConfig(
        d_in=int(activation_meta["d_model"]),
        n_latents=n_latents,
        k=k,
        use_auxk=use_auxk,
        auxk_alpha=auxk_alpha,
        normalize_decoder=normalize_decoder,
        device=str(device_obj),
    )
    seed_everything(train_config.seed)

    with stage_progress("build activation dataloaders", enabled=show_progress):
        train_loader, val_loader, _, _ = build_activation_dataloaders(
            resolved_activation_dir,
            batch_size=train_config.batch_size,
            val_fraction=train_config.val_fraction,
            seed=train_config.seed,
            preprocessing=train_config.preprocessing,
            num_workers=train_config.num_workers,
        )

    output_dir = (
        Path(out_dir).expanduser().resolve()
        if out_dir is not None
        else default_sae_checkpoint_dir(
            str(activation_meta.get("model_type", "unknown")),
            activation_meta.get("checkpoint_step"),
            int(activation_meta.get("layer_idx", 0)),
            str(activation_meta.get("site", "input")),
        )
    )
    model = TopKSAE(sae_config)
    trainer = SAETrainer(
        model,
        sae_config=sae_config,
        train_config=train_config,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        device=device_obj,
        activation_metadata=activation_meta,
        activation_dir=resolved_activation_dir,
        accelerator=accelerator,
        show_progress=show_progress if accelerator is None else show_progress and accelerator.is_main_process,
    )
    if accelerator is not None:
        with stage_progress("prepare accelerate objects", enabled=show_progress and accelerator.is_main_process):
            trainer.model, trainer.optimizer, trainer.train_loader, trainer.val_loader = accelerator.prepare(
                trainer.model,
                trainer.optimizer,
                trainer.train_loader,
                trainer.val_loader,
            )
        trainer.device = accelerator.device
    else:
        trainer.model = trainer.model.to(device_obj)
    return trainer.train()


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
    accelerator = Accelerator()

    if not accelerator.is_main_process:
        logging.getLogger().setLevel(logging.WARNING)

    try:
        summary = train_sae_from_activation_dir(
            args.activation_dir,
            n_latents=args.n_latents,
            k=args.k,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            lr=args.lr,
            out_dir=args.out_dir or None,
            device=args.device,
            use_auxk=args.use_auxk,
            normalize_decoder=args.normalize_decoder,
            input_centering=args.input_centering,
            input_norm=args.input_norm,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            checkpoint_interval=args.checkpoint_interval,
            val_fraction=args.val_fraction,
            seed=args.seed,
            num_workers=args.num_workers,
            auxk_alpha=args.auxk_alpha,
            accelerator=accelerator,
            show_progress=accelerator.is_main_process,
        )
    except Exception as error:
        logger.error("%s", error)
        return 1

    if accelerator.is_main_process:
        logger.info("training finished: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
