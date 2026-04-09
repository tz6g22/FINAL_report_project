"""Run a minimal TinyStories smoke test for StandardGPT and AttnResGPT."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from .config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from .data import build_dataloaders
from .evaluate import load_checkpoint, resolve_device, run_evaluation
from .train import train_model
from .visualize import plot_logs


def evaluate_checkpoint(checkpoint_path: str, eval_batches: int, device_name: str = "auto") -> dict[str, float]:
    device = resolve_device(device_name)
    model, experiment, _ = load_checkpoint(checkpoint_path, device=device)
    _, val_loader = build_dataloaders(
        model_config=experiment.model,
        data_config=experiment.data,
        batch_size=experiment.train.batch_size,
        num_workers=experiment.train.num_workers,
        seed=experiment.train.seed,
    )
    return run_evaluation(model, val_loader, device=device, max_batches=eval_batches)


def build_smoke_experiment(args: argparse.Namespace, out_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            model_type="standard",
            vocab_size=50257,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=0.0,
        ),
        data=DataConfig(
            dataset_type="tinystories",
            hf_dataset_name=args.dataset_name,
            text_field="text",
            tokenizer_name=args.tokenizer_name,
            train_texts=args.train_texts,
            val_texts=args.val_texts,
            block_stride=args.block_stride,
        ),
        train=TrainConfig(
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            eval_interval=args.eval_interval,
            checkpoint_interval=args.checkpoint_interval,
            learning_rate=args.learning_rate,
            eval_batches=args.eval_batches,
            seed=args.seed,
            device=args.device,
            num_workers=args.num_workers,
            out_dir=str(out_dir / "standard"),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TinyStories smoke test for StandardGPT and AttnResGPT.")
    parser.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--out_dir", type=str, default="mini_gpt_attnres_runs/tinystories_smoke")
    parser.add_argument("--train_texts", type=int, default=2000)
    parser.add_argument("--val_texts", type=int, default=200)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--block_stride", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--eval_batches", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(args.out_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    base_experiment = build_smoke_experiment(args, root_dir)

    # Data/tokenizer smoke check before training.
    train_loader, val_loader = build_dataloaders(
        model_config=base_experiment.model,
        data_config=base_experiment.data,
        batch_size=base_experiment.train.batch_size,
        num_workers=base_experiment.train.num_workers,
        seed=base_experiment.train.seed,
    )
    first_batch_inputs, first_batch_labels = next(iter(train_loader))
    dataloader_smoke = {
        "train_batch_input_shape": list(first_batch_inputs.shape),
        "train_batch_label_shape": list(first_batch_labels.shape),
        "val_num_batches": len(val_loader),
        "vocab_size_used": base_experiment.model.vocab_size,
        "input_max_token_id": int(first_batch_inputs.max().item()),
        "label_max_token_id": int(first_batch_labels.max().item()),
    }

    standard_experiment = deepcopy(base_experiment)
    standard_experiment.model.model_type = "standard"
    standard_experiment.train.out_dir = str(root_dir / "standard")
    standard_summary = train_model(standard_experiment)

    attnres_experiment = deepcopy(base_experiment)
    attnres_experiment.model.model_type = "attnres"
    attnres_experiment.train.out_dir = str(root_dir / "attnres")
    attnres_summary = train_model(attnres_experiment)

    standard_eval = evaluate_checkpoint(
        checkpoint_path=standard_summary["checkpoint_path"],
        eval_batches=args.eval_batches,
        device_name=args.device,
    )
    attnres_eval = evaluate_checkpoint(
        checkpoint_path=attnres_summary["checkpoint_path"],
        eval_batches=args.eval_batches,
        device_name=args.device,
    )
    (root_dir / "standard_eval.json").write_text(json.dumps(standard_eval, indent=2), encoding="utf-8")
    (root_dir / "attnres_eval.json").write_text(json.dumps(attnres_eval, indent=2), encoding="utf-8")

    compare_plot = plot_logs(
        log_paths=[standard_summary["log_path"], attnres_summary["log_path"]],
        labels=["StandardGPT", "AttnResGPT"],
        output_path=root_dir / "comparison_val_loss.png",
        compare_split="val",
        title="TinyStories Smoke Validation Loss",
    )
    standard_plot = plot_logs(
        log_paths=[standard_summary["log_path"]],
        labels=["StandardGPT"],
        output_path=root_dir / "standard_loss.png",
        title="TinyStories StandardGPT Loss",
    )
    attnres_plot = plot_logs(
        log_paths=[attnres_summary["log_path"]],
        labels=["AttnResGPT"],
        output_path=root_dir / "attnres_loss.png",
        title="TinyStories AttnResGPT Loss",
    )

    summary = {
        "dataset": {
            "name": args.dataset_name,
            "tokenizer": args.tokenizer_name,
            "train_texts": args.train_texts,
            "val_texts": args.val_texts,
        },
        "dataloader_smoke": dataloader_smoke,
        "standard": standard_summary,
        "attnres": attnres_summary,
        "reloaded_checkpoint_eval": {
            "standard": standard_eval,
            "attnres": attnres_eval,
        },
        "plots": {
            "standard_loss": str(standard_plot),
            "attnres_loss": str(attnres_plot),
            "comparison_val_loss": str(compare_plot),
        },
    }
    (root_dir / "tinystories_smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
