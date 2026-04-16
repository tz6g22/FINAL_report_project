"""Smoke test for SAE held-out mem/nonmem analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data import build_dataloaders
from scripts.run_sae_visualize import main as run_sae_visualize_main
from stream_analysis.sae.analysis import run_sae_mem_analysis
from stream_analysis.sae.config import SAEConfig, SAEExtractConfig, SAETrainConfig
from stream_analysis.sae.data import ActivationShardDataset, build_activation_dataloaders
from stream_analysis.sae.extract import extract_activation_shards
from stream_analysis.sae.model import TopKSAE
from stream_analysis.sae.train import SAETrainer
from toygpt2.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from toygpt2.model import build_model


def _make_experiment(root: Path, *, model_type: str = "standard") -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            model_type=model_type,
            vocab_size=32,
            block_size=8,
            n_layer=2,
            n_head=2,
            n_embd=16,
            dropout=0.0,
        ),
        data=DataConfig(
            dataset_type="repeated_pattern",
            train_size=12,
            val_size=6,
            pattern_length=4,
            block_stride=8,
        ),
        train=TrainConfig(
            batch_size=2,
            max_steps=1,
            seed=17,
            num_workers=0,
            out_dir=str(root / "toy_run"),
        ),
    )


def _save_tiny_checkpoint(root: Path, *, model_type: str = "standard") -> tuple[Path, ExperimentConfig]:
    experiment = _make_experiment(root, model_type=model_type)
    model = build_model(model_type, experiment.model).eval()
    checkpoint = {
        "step": 11,
        "model_type": model_type,
        "model_config": experiment.to_dict()["model"],
        "data_config": experiment.to_dict()["data"],
        "train_config": experiment.to_dict()["train"],
        "model_state": model.state_dict(),
        "optimizer_state": {},
    }
    checkpoint_path = root / f"ckpt_{model_type}.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path, experiment


def _build_labels_source(path: Path, experiment: ExperimentConfig) -> Path:
    train_loader, val_loader = build_dataloaders(
        model_config=experiment.model,
        data_config=experiment.data,
        batch_size=2,
        num_workers=0,
        seed=experiment.train.seed,
        verbose=False,
    )
    train_inputs, train_labels = next(iter(train_loader))
    val_inputs, val_labels = next(iter(val_loader))
    input_ids = torch.cat([train_inputs[:2], val_inputs[:2]], dim=0)
    labels = torch.cat([train_labels[:2], val_labels[:2]], dim=0)
    payload = {
        "meta": {
            "dataset_type": experiment.data.dataset_type,
            "tokenizer_name": None,
        },
        "sample_ids": [f"sample_{idx:02d}" for idx in range(int(input_ids.shape[0]))],
        "group_labels": ["general_train", "general_train", "general_val", "general_val"],
        "input_ids": input_ids,
        "labels": labels,
        "target_positions": torch.full((int(input_ids.shape[0]),), experiment.model.block_size - 1, dtype=torch.long),
        "texts": None,
    }
    torch.save(payload, path)
    return path


def test_sae_mem_analysis_smoke(tmp_path: Path) -> None:
    checkpoint_path, experiment = _save_tiny_checkpoint(tmp_path, model_type="standard")

    train_activation_dir = tmp_path / "activations_train"
    extract_activation_shards(
        SAEExtractConfig(
            model_type="standard",
            checkpoint_path=str(checkpoint_path),
            layer_idx=0,
            site="input",
            dataset_split="train",
            max_tokens=32,
            out_dir=str(train_activation_dir),
            batch_size=2,
            device="cpu",
            shard_size_tokens=10,
            overwrite=True,
        )
    )
    val_activation_dir = tmp_path / "activations_val"
    extract_activation_shards(
        SAEExtractConfig(
            model_type="standard",
            checkpoint_path=str(checkpoint_path),
            layer_idx=0,
            site="input",
            dataset_split="val",
            max_tokens=24,
            out_dir=str(val_activation_dir),
            batch_size=2,
            device="cpu",
            shard_size_tokens=10,
            overwrite=True,
        )
    )

    dataset = ActivationShardDataset(train_activation_dir)
    train_loader, val_loader, _, _ = build_activation_dataloaders(
        train_activation_dir,
        batch_size=8,
        val_fraction=0.25,
        seed=0,
        preprocessing="mean-center",
        num_workers=0,
    )
    sae_config = SAEConfig(
        d_in=dataset.d_in,
        n_latents=32,
        k=4,
        use_auxk=True,
        auxk_alpha=0.01,
        normalize_decoder=True,
        device="cpu",
    )
    model = TopKSAE(sae_config)
    train_config = SAETrainConfig(
        batch_size=8,
        num_steps=3,
        lr=1e-3,
        eval_interval=1,
        log_interval=1,
        checkpoint_interval=2,
        val_fraction=0.25,
        seed=0,
        preprocessing="mean-center",
        device="cpu",
    )
    output_dir = tmp_path / "sae_ckpts"
    trainer = SAETrainer(
        model,
        sae_config=sae_config,
        train_config=train_config,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        device=torch.device("cpu"),
        activation_metadata=dataset.metadata,
        activation_dir=train_activation_dir,
    )
    trainer.train()

    labels_source = _build_labels_source(tmp_path / "analysis_set.pt", experiment)
    analysis_dir = tmp_path / "sae_analysis"
    result = run_sae_mem_analysis(
        output_dir / "best.pt",
        val_activation_dir,
        labels_source,
        batch_size=2,
        device="cpu",
        out_dir=analysis_dir,
        topk_examples=2,
    )

    assert Path(result["eval_summary_path"]).is_file()
    assert Path(result["feature_stats_path"]).is_file()
    assert Path(result["feature_auc_path"]).is_file()
    assert Path(result["top_examples_path"]).is_file()
    assert Path(result["feature_payload_path"]).is_file()

    with Path(result["eval_summary_path"]).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert "summary" in payload
    assert "feature_summary" in payload
    assert "decoder_overlap" in payload
    assert "coactivation" in payload

    visualize_out_dir = tmp_path / "viz"
    exit_code = run_sae_visualize_main(
        [
            "--analysis-dir",
            str(analysis_dir),
            "--out-dir",
            str(visualize_out_dir),
        ]
    )
    assert exit_code == 0
    assert (visualize_out_dir / "comparison_metrics.csv").is_file()
    assert any(path.name.startswith("decoder_overlap_heatmap_") for path in (visualize_out_dir / "figures").glob("*.png"))
    assert any(path.name.startswith("coactivation_heatmap_") for path in (visualize_out_dir / "figures").glob("*.png"))
