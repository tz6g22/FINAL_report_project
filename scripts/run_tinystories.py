"""Run formal TinyStories training for StandardGPT and/or AttnResGPT."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from toygpt2.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from scripts.evaluate import load_checkpoint, resolve_device, run_evaluation
from scripts.train import train_model


def evaluate_checkpoint(checkpoint_path: str, eval_batches: int | None, device_name: str = "auto") -> dict[str, float]:
    device = resolve_device(device_name)
    model, experiment, _ = load_checkpoint(checkpoint_path, device=device)
    from data.data import build_dataloaders

    _, val_loader = build_dataloaders(
        model_config=experiment.model,
        data_config=experiment.data,
        batch_size=experiment.train.batch_size,
        num_workers=experiment.train.num_workers,
        seed=experiment.train.seed,
    )
    return run_evaluation(model, val_loader, device=device, max_batches=eval_batches)


def build_experiment(args: argparse.Namespace, out_dir: Path) -> ExperimentConfig:
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
    parser = argparse.ArgumentParser(description="Formal TinyStories training for StandardGPT and AttnResGPT.")
    parser.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--out_dir", type=str, default="toygpt2_runs/tinystories")
    parser.add_argument("--train_texts", type=int, default=None, help="Optional cap; default uses full train split.")
    parser.add_argument("--val_texts", type=int, default=None, help="Optional cap; default uses full validation split.")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--block_stride", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument("--eval_batches", type=int, default=None, help="Optional cap; default evaluates full validation split.")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_layer", type=int, default=8)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--run_mode",
        type=str,
        default="both",
        choices=["both", "standard", "attnres"],
        help="Choose which model(s) to train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from scripts.visualize import plot_logs

    root_dir = Path(args.out_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    base_experiment = build_experiment(args, root_dir)
    models_to_run = ["standard", "attnres"] if args.run_mode == "both" else [args.run_mode]

    label_by_model = {"standard": "StandardGPT", "attnres": "AttnResGPT"}
    train_summaries: dict[str, dict[str, object]] = {}
    eval_metrics_by_model: dict[str, dict[str, float]] = {}
    plots: dict[str, str] = {}

    for model_type in models_to_run:
        print(f"[runner] preparing model={model_type}", flush=True)
        experiment = deepcopy(base_experiment)
        experiment.model.model_type = model_type
        experiment.train.out_dir = str(root_dir / model_type)
        train_summary = train_model(experiment)
        train_summaries[model_type] = train_summary

        print(f"[runner] evaluating model={model_type}", flush=True)
        eval_metrics = evaluate_checkpoint(
            checkpoint_path=train_summary["checkpoint_path"],
            eval_batches=args.eval_batches,
            device_name=args.device,
        )
        eval_metrics_by_model[model_type] = eval_metrics
        (root_dir / f"{model_type}_eval.json").write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")

        loss_plot = plot_logs(
            log_paths=[train_summary["log_path"]],
            labels=[label_by_model[model_type]],
            output_path=root_dir / f"{model_type}_loss.png",
            title=f"TinyStories {label_by_model[model_type]} Loss",
        )
        plots[f"{model_type}_loss"] = str(loss_plot)

    if len(models_to_run) == 2:
        compare_plot = plot_logs(
            log_paths=[train_summaries["standard"]["log_path"], train_summaries["attnres"]["log_path"]],
            labels=["StandardGPT", "AttnResGPT"],
            output_path=root_dir / "comparison_val_loss.png",
            compare_split="val",
            title="TinyStories Validation Loss",
        )
        plots["comparison_val_loss"] = str(compare_plot)

    summary = {
        "dataset": {
            "name": args.dataset_name,
            "tokenizer": args.tokenizer_name,
            "train_texts": args.train_texts,
            "val_texts": args.val_texts,
        },
        "run_mode": args.run_mode,
        "models_ran": models_to_run,
        "reloaded_checkpoint_eval": eval_metrics_by_model,
        "plots": plots,
    }
    if "standard" in train_summaries:
        summary["standard"] = train_summaries["standard"]
    if "attnres" in train_summaries:
        summary["attnres"] = train_summaries["attnres"]

    (root_dir / "tinystories_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
