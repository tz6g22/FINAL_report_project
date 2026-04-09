"""Minimal baseline-vs-AttnRes comparison on toy data."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from .config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from .data import build_dataloaders
from .evaluate import load_checkpoint, resolve_device, run_evaluation
from .train import train_model
from .visualize import plot_logs


def evaluate_checkpoint(checkpoint_path: str, device_name: str, eval_batches: int) -> dict[str, float]:
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


def main() -> None:
    root_dir = Path("mini_gpt_attnres_runs/demo_compare")
    experiment = ExperimentConfig(
        model=ModelConfig(
            vocab_size=48,
            block_size=32,
            n_layer=2,
            n_head=2,
            n_embd=64,
            dropout=0.0,
        ),
        data=DataConfig(
            dataset_type="repeated_pattern",
            train_size=256,
            val_size=64,
            pattern_length=8,
        ),
        train=TrainConfig(
            batch_size=16,
            max_steps=30,
            eval_interval=10,
            checkpoint_interval=15,
            learning_rate=3e-4,
            eval_batches=4,
            seed=1234,
            device="auto",
            out_dir=str(root_dir / "standard"),
        ),
    )

    standard_experiment = deepcopy(experiment)
    standard_experiment.model.model_type = "standard"
    standard_experiment.train.out_dir = str(root_dir / "standard")
    standard_summary = train_model(standard_experiment)

    attnres_experiment = deepcopy(experiment)
    attnres_experiment.model.model_type = "attnres"
    attnres_experiment.train.out_dir = str(root_dir / "attnres")
    attnres_summary = train_model(attnres_experiment)

    standard_eval = evaluate_checkpoint(
        checkpoint_path=standard_summary["checkpoint_path"],
        device_name=standard_experiment.train.device,
        eval_batches=standard_experiment.train.eval_batches,
    )
    attnres_eval = evaluate_checkpoint(
        checkpoint_path=attnres_summary["checkpoint_path"],
        device_name=attnres_experiment.train.device,
        eval_batches=attnres_experiment.train.eval_batches,
    )
    (root_dir / "standard_eval.json").write_text(json.dumps(standard_eval, indent=2), encoding="utf-8")
    (root_dir / "attnres_eval.json").write_text(json.dumps(attnres_eval, indent=2), encoding="utf-8")

    standard_plot = plot_logs(
        log_paths=[standard_summary["log_path"]],
        output_path=root_dir / "standard_loss.png",
        labels=["StandardGPT"],
        title="StandardGPT Loss Curves",
    )
    attnres_plot = plot_logs(
        log_paths=[attnres_summary["log_path"]],
        output_path=root_dir / "attnres_loss.png",
        labels=["AttnResGPT"],
        title="AttnResGPT Loss Curves",
    )
    compare_plot = plot_logs(
        log_paths=[standard_summary["log_path"], attnres_summary["log_path"]],
        labels=["StandardGPT", "AttnResGPT"],
        output_path=root_dir / "comparison_val_loss.png",
        compare_split="val",
        title="Validation Loss Comparison",
    )

    print("Final comparison summary:")
    print(
        json.dumps(
            {
                "standard": standard_summary,
                "attnres": attnres_summary,
                "reloaded_checkpoint_eval": {
                    "standard": standard_eval,
                    "attnres": attnres_eval,
                },
                "plots": {
                    "standard_loss": standard_plot,
                    "attnres_loss": attnres_plot,
                    "comparison_val_loss": compare_plot,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
