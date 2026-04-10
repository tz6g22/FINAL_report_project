"""Minimal training loop for StandardGPT and AttnResGPT."""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterator, Tuple

import torch

from .config import ExperimentConfig, default_experiment
from .data import build_dataloaders
from .evaluate import resolve_device, run_evaluation
from .model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cycle_dataloader(dataloader: torch.utils.data.DataLoader) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in dataloader:
            yield batch


def append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


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
    experiment_dict = experiment.to_dict()
    checkpoint = {
        "step": step,
        "model_type": model_type,
        "model_config": experiment_dict["model"],
        "data_config": experiment_dict["data"],
        "train_config": experiment_dict["train"],
        "model_state": model.state_dict(),
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
    if model_type is not None:
        experiment.model.model_type = model_type
    model_type = experiment.model.model_type
    if out_dir is not None:
        experiment.train.out_dir = out_dir

    set_seed(experiment.train.seed)
    device = resolve_device(experiment.train.device)

    output_dir = Path(experiment.train.out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "metrics.jsonl"
    log_path.write_text("", encoding="utf-8")

    train_loader, val_loader = build_dataloaders(
        model_config=experiment.model,
        data_config=experiment.data,
        batch_size=experiment.train.batch_size,
        num_workers=experiment.train.num_workers,
        seed=experiment.train.seed,
    )
    experiment.save_json(output_dir / "config.json")
    train_batches = cycle_dataloader(train_loader)

    model = build_model(model_type, experiment.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=experiment.train.learning_rate,
        weight_decay=experiment.train.weight_decay,
    )

    print(
        f"[train][{model_type}] start training: steps={experiment.train.max_steps} "
        f"batch_size={experiment.train.batch_size} device={device}",
        flush=True,
    )

    final_train_loss = None
    final_val_metrics: Dict[str, float] | None = None
    for step in range(1, experiment.train.max_steps + 1):
        model.train()
        inputs, targets = next(train_batches)
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs, targets=targets)
        loss = outputs["loss"]
        logits = outputs["logits"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), experiment.train.grad_clip)
        optimizer.step()

        final_train_loss = float(loss.item())
        train_acc = batch_accuracy(logits, targets)

        append_jsonl(
            log_path,
            {
                "step": step,
                "split": "train",
                "loss": final_train_loss,
                "accuracy": train_acc,
            },
        )

        message = (
            f"[train][{model_type}] step {step}/{experiment.train.max_steps} "
            f"loss={final_train_loss:.6f} acc={train_acc:.4f}"
        )
        print(message, flush=True)

        if step % experiment.train.eval_interval == 0 or step == experiment.train.max_steps:
            final_val_metrics = run_evaluation(
                model,
                val_loader,
                device=device,
                max_batches=experiment.train.eval_batches,
            )
            append_jsonl(
                log_path,
                {
                    "step": step,
                    "split": "val",
                    **final_val_metrics,
                },
            )

        if step % experiment.train.checkpoint_interval == 0 or step == experiment.train.max_steps:
            save_checkpoint(
                path=output_dir / f"ckpt_step_{step:06d}.pt",
                model=model,
                optimizer=optimizer,
                experiment=experiment,
                model_type=model_type,
                step=step,
            )

    save_checkpoint(
        path=output_dir / "ckpt_last.pt",
        model=model,
        optimizer=optimizer,
        experiment=experiment,
        model_type=model_type,
        step=experiment.train.max_steps,
    )

    summary = {
        "model_type": model_type,
        "device": str(device),
        "num_parameters": model.num_parameters(),
        "final_train_loss": final_train_loss,
        "final_val_loss": None if final_val_metrics is None else final_val_metrics["loss"],
        "final_val_perplexity": None if final_val_metrics is None else final_val_metrics["perplexity"],
        "final_val_accuracy": None if final_val_metrics is None else final_val_metrics["accuracy"],
        "log_path": str(log_path),
        "checkpoint_path": str(output_dir / "ckpt_last.pt"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    val_loss = summary["final_val_loss"]
    val_loss_str = "None" if val_loss is None else f"{val_loss:.6f}"
    print(
        f"[train][{model_type}] finished training: "
        f"final_train_loss={summary['final_train_loss']:.6f} "
        f"final_val_loss={val_loss_str}",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train StandardGPT or AttnResGPT on toy data.")
    parser.add_argument("--model_type", type=str, default="standard", choices=["standard", "attnres"])
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file.")
    parser.add_argument("--out_dir", type=str, default=None, help="Optional run output directory.")
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, or auto.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max training steps.")
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

    summary = train_model(experiment)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
