"""Minimal smoke tests for SAE extract / train / eval."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stream_analysis.sae.config import SAEConfig, SAEExtractConfig, SAETrainConfig
from stream_analysis.path_utils import format_project_path, project_root, resolve_project_path
from stream_analysis.sae.data import ActivationShardDataset, build_activation_dataloaders
from stream_analysis.sae.eval import SAEEvaluator, load_sae_checkpoint, save_evaluation_results
from stream_analysis.sae.extract import extract_activation_shards
from stream_analysis.sae.model import TopKSAE
from stream_analysis.sae.train import SAETrainer
from toygpt2.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from toygpt2.model import build_model


def test_checkpoint_path_resolution_uses_repo_relative_style() -> None:
    relative = Path("toygpt2_runs/tinystories_dual/standard/ckpt_standard_last.pt")
    resolved = resolve_project_path(relative)
    assert resolved == project_root() / relative
    assert format_project_path(relative) == str(relative)


def _save_tiny_checkpoint(root: Path, *, model_type: str = "standard") -> Path:
    experiment = ExperimentConfig(
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
    return checkpoint_path


def test_sae_extract_train_eval_smoke(tmp_path: Path) -> None:
    checkpoint_path = _save_tiny_checkpoint(tmp_path, model_type="standard")
    activation_dir = tmp_path / "activations"
    extract_config = SAEExtractConfig(
        model_type="standard",
        checkpoint_path=str(checkpoint_path),
        layer_idx=0,
        site="input",
        dataset_split="train",
        max_tokens=32,
        out_dir=str(activation_dir),
        batch_size=2,
        device="cpu",
        shard_size_tokens=10,
        overwrite=True,
    )
    summary = extract_activation_shards(extract_config)
    assert summary.num_tokens == 32
    assert Path(summary.meta_path).is_file()

    dataset = ActivationShardDataset(activation_dir)
    assert len(dataset) == 32
    assert dataset[0].shape == (16,)

    train_loader, val_loader, _, _ = build_activation_dataloaders(
        activation_dir,
        batch_size=8,
        val_fraction=0.25,
        seed=0,
        preprocessing="mean-center",
        num_workers=0,
    )
    sae_config = SAEConfig(
        d_in=16,
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
        activation_dir=activation_dir,
    )
    train_summary = trainer.train()
    assert Path(train_summary["metrics_path"]).is_file()
    assert (output_dir / "best.pt").is_file()
    assert (output_dir / "last.pt").is_file()

    loaded_model, loaded_cfg, _ = load_sae_checkpoint(output_dir / "best.pt", device=torch.device("cpu"))
    evaluator = SAEEvaluator(loaded_model, loaded_cfg, device=torch.device("cpu"))
    metrics = evaluator.evaluate_activation_dir(
        activation_dir,
        batch_size=8,
        preprocessing="mean-center",
        num_workers=0,
    )
    assert metrics["recon_mse"] >= 0.0
    assert 0.0 <= metrics["dead_latent_frac"] <= 1.0

    metrics_path = save_evaluation_results(metrics, tmp_path / "eval")
    assert metrics_path.is_file()
