"""Checkpoint loading and shared evaluation utilities."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict

import torch

from .config import ExperimentConfig
from .data import build_dataloaders
from .model import build_model


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@torch.no_grad()
def run_evaluation(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> Dict[str, float]:
    """Compute loss, perplexity, and token accuracy."""

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs, targets=targets)
        loss = outputs["loss"]
        logits = outputs["logits"]

        total_loss += float(loss.item()) * targets.numel()
        total_correct += int((logits.argmax(dim=-1) == targets).sum().item())
        total_tokens += targets.numel()

    avg_loss = total_loss / max(1, total_tokens)
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    accuracy = total_correct / max(1, total_tokens)
    return {"loss": avg_loss, "perplexity": perplexity, "accuracy": accuracy}


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, ExperimentConfig, Dict[str, object]]:
    """Rebuild a model from a saved checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    experiment = ExperimentConfig.from_dict(
        {
            "model": checkpoint["model_config"],
            "data": checkpoint["data_config"],
            "train": checkpoint["train_config"],
        }
    )
    model = build_model(checkpoint["model_type"], experiment.model).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, experiment, checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved StandardGPT or AttnResGPT checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the saved checkpoint.")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto.")
    parser.add_argument("--eval_batches", type=int, default=None, help="Optional limit for evaluation batches.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Dataset split to evaluate.")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to save metrics as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, experiment, checkpoint = load_checkpoint(args.checkpoint, device)

    train_loader, val_loader = build_dataloaders(
        model_config=experiment.model,
        data_config=experiment.data,
        batch_size=experiment.train.batch_size,
        num_workers=experiment.train.num_workers,
        seed=experiment.train.seed,
    )
    dataloader = train_loader if args.split == "train" else val_loader
    metrics = run_evaluation(model, dataloader, device=device, max_batches=args.eval_batches)

    print(f"checkpoint: {args.checkpoint}")
    print(f"model_type: {checkpoint['model_type']}")
    print(f"split: {args.split}")
    print(json.dumps(metrics, indent=2))

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()
