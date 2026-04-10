"""Minimal training loop for StandardGPT and AttnResGPT."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from .config import ExperimentConfig, default_experiment
from .data import build_dataloaders
from .evaluate import resolve_device, run_evaluation
from .model import build_model


@dataclass
class DistributedContext:
    """Runtime distributed context for one torchrun worker."""

    enabled: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str | None = None

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _init_distributed_if_needed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext(enabled=False)

    if not torch.cuda.is_available():
        raise RuntimeError("DDP requested via WORLD_SIZE>1, but CUDA is unavailable.")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return DistributedContext(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend="nccl",
    )


def _cleanup_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def _barrier(context: DistributedContext) -> None:
    if context.enabled:
        dist.barrier()


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    path = Path(path)
    payload_line = json.dumps(payload) + "\n"

    # Some distributed filesystems can transiently report ENOENT after mkdir.
    # Retry once after recreating the parent to avoid rank-0 log-write crashes.
    last_error: FileNotFoundError | None = None
    for _ in range(2):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(payload_line)
            return
        except FileNotFoundError as error:
            last_error = error
            time.sleep(0.05)

    raise FileNotFoundError(f"Failed to append JSONL to '{path}' after ensuring parent directory.") from last_error


def batch_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == targets).float().mean().item())


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    experiment: ExperimentConfig,
    model_type: str,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = _unwrap_model(model)
    experiment_dict = experiment.to_dict()
    checkpoint = {
        "step": step,
        "model_type": model_type,
        "model_config": experiment_dict["model"],
        "data_config": experiment_dict["data"],
        "train_config": experiment_dict["train"],
        "model_state": raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)


def train_model(
    experiment: ExperimentConfig,
    model_type: str | None = None,
    out_dir: str | None = None,
) -> Dict[str, object]:
    """Train one model variant and return a summary dict."""

    experiment = deepcopy(experiment)
    dist_ctx = _init_distributed_if_needed()

    if model_type is not None:
        experiment.model.model_type = model_type
    model_type = experiment.model.model_type
    if out_dir is not None:
        experiment.train.out_dir = out_dir

    try:
        set_seed(experiment.train.seed + dist_ctx.rank)
        if dist_ctx.enabled:
            device = torch.device("cuda", dist_ctx.local_rank)
        else:
            device = resolve_device(experiment.train.device)

        output_dir = Path(experiment.train.out_dir).expanduser().resolve()
        log_path = output_dir / f"metrics_{model_type}.jsonl"
        summary_path = output_dir / f"summary_{model_type}.json"
        checkpoint_last_path = output_dir / f"ckpt_{model_type}_last.pt"
        if dist_ctx.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path.write_text("", encoding="utf-8")
            experiment.save_json(output_dir / "config.json")
        _barrier(dist_ctx)

        train_loader, val_loader = build_dataloaders(
            model_config=experiment.model,
            data_config=experiment.data,
            batch_size=experiment.train.batch_size,
            num_workers=experiment.train.num_workers,
            seed=experiment.train.seed,
            distributed=dist_ctx.enabled,
            rank=dist_ctx.rank,
            world_size=dist_ctx.world_size,
            verbose=dist_ctx.is_main_process,
        )

        model = build_model(model_type, experiment.model).to(device)
        if dist_ctx.enabled:
            model = DDP(
                model,
                device_ids=[dist_ctx.local_rank],
                output_device=dist_ctx.local_rank,
                find_unused_parameters=False,
            )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=experiment.train.learning_rate,
            weight_decay=experiment.train.weight_decay,
        )

        steps_per_epoch = len(train_loader)
        if steps_per_epoch <= 0:
            raise RuntimeError("train_loader has zero steps per epoch; cannot run training.")
        if experiment.train.eval_interval <= 0:
            raise ValueError("eval_interval must be > 0.")
        if experiment.train.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be > 0.")
        effective_global_batch = experiment.train.batch_size * dist_ctx.world_size
        current_device = torch.cuda.current_device() if device.type == "cuda" else None
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<not-set>")

        if dist_ctx.is_main_process:
            print(
                f"[{model_type}] start training: max_steps={experiment.train.max_steps} "
                f"batch_size_per_gpu={experiment.train.batch_size} global_batch={effective_global_batch} "
                f"device={device} world_size={dist_ctx.world_size}",
                flush=True,
            )
            print(
                f"[{model_type}] ddp_mapping rank={dist_ctx.rank} local_rank={dist_ctx.local_rank} "
                f"world_size={dist_ctx.world_size} current_device={current_device} "
                f"CUDA_VISIBLE_DEVICES={visible_devices}",
                flush=True,
            )

        final_train_loss = None
        final_val_metrics: Dict[str, float] | None = None
        global_step = 0
        epoch = 0

        while global_step < experiment.train.max_steps:
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            for step_in_epoch, (inputs, targets) in enumerate(train_loader, start=1):
                if global_step >= experiment.train.max_steps:
                    break
                step_start = time.perf_counter()

                model.train()
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs, targets=targets)
                loss = outputs["loss"]
                logits = outputs["logits"]
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), experiment.train.grad_clip))
                optimizer.step()

                global_step += 1
                final_train_loss = float(loss.item())
                train_acc = batch_accuracy(logits, targets)
                lr = float(optimizer.param_groups[0]["lr"])
                step_time = time.perf_counter() - step_start

                if dist_ctx.is_main_process:
                    append_jsonl(
                        log_path,
                        {
                            "epoch": epoch,
                            "step": step_in_epoch,
                            "steps_per_epoch": steps_per_epoch,
                            "global_step": global_step,
                            "split": "train",
                            "loss": final_train_loss,
                            "accuracy": train_acc,
                            "lr": lr,
                            "step_time": step_time,
                            "grad_norm": grad_norm,
                            "effective_world_size": dist_ctx.world_size,
                            "per_gpu_batch_size": experiment.train.batch_size,
                        },
                    )
                    print(
                        f"[{model_type}] epoch={epoch} step={step_in_epoch}/{steps_per_epoch} "
                        f"global_step={global_step} loss={final_train_loss:.6f} lr={lr:.6e} "
                        f"step_time={step_time:.3f}s grad_norm={grad_norm:.4f} "
                        f"per_gpu_batch_size={experiment.train.batch_size} "
                        f"effective_world_size={dist_ctx.world_size} rank=0",
                        flush=True,
                    )

                should_eval = global_step % experiment.train.eval_interval == 0 or global_step == experiment.train.max_steps
                if should_eval:
                    _barrier(dist_ctx)
                    if dist_ctx.is_main_process:
                        eval_model = _unwrap_model(model)
                        final_val_metrics = run_evaluation(
                            eval_model,
                            val_loader,
                            device=device,
                            max_batches=experiment.train.eval_batches,
                        )
                        append_jsonl(
                            log_path,
                            {
                                "epoch": epoch,
                                "global_step": global_step,
                                "split": "val",
                                **final_val_metrics,
                            },
                        )
                        print(
                            f"[{model_type}] eval global_step={global_step} "
                            f"val_loss={final_val_metrics['loss']:.6f} "
                            f"val_ppl={final_val_metrics['perplexity']:.4f} "
                            f"val_acc={final_val_metrics['accuracy']:.4f}",
                            flush=True,
                        )
                    _barrier(dist_ctx)

                should_save = (
                    global_step % experiment.train.checkpoint_interval == 0
                    or global_step == experiment.train.max_steps
                )
                if should_save:
                    _barrier(dist_ctx)
                    if dist_ctx.is_main_process:
                        save_checkpoint(
                            path=output_dir / f"ckpt_{model_type}_step_{global_step:06d}.pt",
                            model=model,
                            optimizer=optimizer,
                            experiment=experiment,
                            model_type=model_type,
                            step=global_step,
                        )
                    _barrier(dist_ctx)
            epoch += 1

        _barrier(dist_ctx)
        if dist_ctx.is_main_process:
            save_checkpoint(
                path=checkpoint_last_path,
                model=model,
                optimizer=optimizer,
                experiment=experiment,
                model_type=model_type,
                step=global_step,
            )

            raw_model = _unwrap_model(model)
            summary = {
                "model_type": model_type,
                "device": str(device),
                "world_size": dist_ctx.world_size,
                "distributed_backend": dist_ctx.backend,
                "batch_size_per_gpu": experiment.train.batch_size,
                "global_batch_size": effective_global_batch,
                "num_parameters": raw_model.num_parameters(),
                "final_train_loss": final_train_loss,
                "final_val_loss": None if final_val_metrics is None else final_val_metrics["loss"],
                "final_val_perplexity": None if final_val_metrics is None else final_val_metrics["perplexity"],
                "final_val_accuracy": None if final_val_metrics is None else final_val_metrics["accuracy"],
                "log_path": str(log_path),
                "checkpoint_path": str(checkpoint_last_path),
                "summary_path": str(summary_path),
                "is_main_process": True,
                "rank": dist_ctx.rank,
                "local_rank": dist_ctx.local_rank,
                "current_device": current_device,
            }
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with summary_path.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
            val_loss = summary["final_val_loss"]
            val_loss_str = "None" if val_loss is None else f"{val_loss:.6f}"
            print(
                f"[{model_type}] finished training: "
                f"final_train_loss={summary['final_train_loss']:.6f} "
                f"final_val_loss={val_loss_str}",
                flush=True,
            )
            return summary

        return {
            "model_type": model_type,
            "is_main_process": False,
            "rank": dist_ctx.rank,
            "world_size": dist_ctx.world_size,
        }
    finally:
        _cleanup_distributed(dist_ctx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train StandardGPT or AttnResGPT on toy data.")
    parser.add_argument("--model_type", type=str, default="standard", choices=["standard", "attnres"])
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file.")
    parser.add_argument("--out_dir", type=str, default=None, help="Optional run output directory.")
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, or auto.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max training steps.")
    parser.add_argument("--batch_size", type=int, default=None, help="Per-process batch size (per GPU under DDP).")
    parser.add_argument("--eval_interval", type=int, default=None, help="Validation interval in optimizer steps.")
    parser.add_argument("--checkpoint_interval", type=int, default=None, help="Checkpoint interval in optimizer steps.")
    parser.add_argument("--eval_batches", type=int, default=None, help="Validation batch limit.")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--num_workers", type=int, default=None, help="Dataloader worker count.")
    parser.add_argument("--dataset_type", type=str, default=None, choices=["random", "repeated_pattern", "retrieval", "tinystories"])
    parser.add_argument("--dataset_name", type=str, default=None, help="HF dataset name for TinyStories mode.")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Tokenizer name for TinyStories mode.")
    parser.add_argument(
        "--train_texts",
        type=int,
        default=None,
        help="Optional TinyStories train text cap. Default uses the full train split.",
    )
    parser.add_argument(
        "--val_texts",
        type=int,
        default=None,
        help="Optional TinyStories validation text cap. Default uses the full validation split.",
    )
    parser.add_argument("--block_size", type=int, default=None, help="Model context length.")
    parser.add_argument("--block_stride", type=int, default=None, help="TinyStories block stride.")
    parser.add_argument("--n_layer", type=int, default=None, help="Model layer count.")
    parser.add_argument("--n_head", type=int, default=None, help="Model head count.")
    parser.add_argument("--n_embd", type=int, default=None, help="Model embedding width.")
    parser.add_argument("--train_size", type=int, default=None, help="Synthetic dataset train size.")
    parser.add_argument("--val_size", type=int, default=None, help="Synthetic dataset val size.")
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        dest="local_rank",
        type=int,
        default=None,
        help="Torchrun local rank (kept for compatibility).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = default_experiment(model_type=args.model_type)
    if args.config is not None:
        experiment = ExperimentConfig.load_json(args.config)
        experiment.model.model_type = args.model_type
    if args.out_dir is not None:
        experiment.train.out_dir = args.out_dir
    if args.device is not None:
        experiment.train.device = args.device
    if args.seed is not None:
        experiment.train.seed = args.seed
    if args.max_steps is not None:
        experiment.train.max_steps = args.max_steps
    if args.batch_size is not None:
        experiment.train.batch_size = args.batch_size
    if args.eval_interval is not None:
        experiment.train.eval_interval = args.eval_interval
    if args.checkpoint_interval is not None:
        experiment.train.checkpoint_interval = args.checkpoint_interval
    if args.eval_batches is not None:
        experiment.train.eval_batches = args.eval_batches
    if args.learning_rate is not None:
        experiment.train.learning_rate = args.learning_rate
    if args.num_workers is not None:
        experiment.train.num_workers = args.num_workers

    if args.dataset_type is not None:
        experiment.data.dataset_type = args.dataset_type
    if args.dataset_name is not None:
        experiment.data.hf_dataset_name = args.dataset_name
    if args.tokenizer_name is not None:
        experiment.data.tokenizer_name = args.tokenizer_name
    if args.train_texts is not None:
        experiment.data.train_texts = args.train_texts
    if args.val_texts is not None:
        experiment.data.val_texts = args.val_texts
    if args.block_stride is not None:
        experiment.data.block_stride = args.block_stride
    if args.train_size is not None:
        experiment.data.train_size = args.train_size
    if args.val_size is not None:
        experiment.data.val_size = args.val_size

    if args.block_size is not None:
        experiment.model.block_size = args.block_size
    if args.n_layer is not None:
        experiment.model.n_layer = args.n_layer
    if args.n_head is not None:
        experiment.model.n_head = args.n_head
    if args.n_embd is not None:
        experiment.model.n_embd = args.n_embd

    summary = train_model(experiment)
    if summary.get("is_main_process", True):
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
