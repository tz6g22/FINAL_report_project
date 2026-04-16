"""Smoke test for the one-command dual-model SAE pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stream_analysis.sae.pipeline import run_dual_sae_pipeline
from toygpt2.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from toygpt2.model import build_model


def _save_tiny_checkpoint(root: Path, *, model_type: str) -> Path:
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
            out_dir=str(root / f"toy_run_{model_type}"),
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


def test_dual_sae_pipeline_smoke(tmp_path: Path) -> None:
    standard_checkpoint = _save_tiny_checkpoint(tmp_path, model_type="standard")
    attnres_checkpoint = _save_tiny_checkpoint(tmp_path, model_type="attnres")

    args = argparse.Namespace(
        standard_checkpoint=str(standard_checkpoint),
        attnres_checkpoint=str(attnres_checkpoint),
        layer=0,
        site="input",
        train_max_tokens=32,
        val_max_tokens=16,
        extract_batch_size=2,
        extract_num_workers=0,
        n_latents=32,
        k=4,
        sae_batch_size=8,
        eval_batch_size=8,
        train_num_workers=0,
        num_steps=3,
        lr=1e-3,
        eval_interval=1,
        checkpoint_interval=2,
        val_fraction=0.25,
        seed=0,
        device="cpu",
        normalize_decoder=True,
        input_centering=True,
        input_norm=False,
        use_auxk=True,
        auxk_alpha=0.01,
        out_dir=str(tmp_path / "dual_pipeline"),
        overwrite=True,
    )

    summary = run_dual_sae_pipeline(args)
    summary_path = Path(summary["output_root"]) / "pipeline_summary.json"
    assert summary_path.is_file()

    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert sorted(payload["models"].keys()) == ["attnres", "standard"]
    for model_type in ("standard", "attnres"):
        model_payload = payload["models"][model_type]
        assert Path(model_payload["train_activation_dir"]).is_dir()
        assert Path(model_payload["val_activation_dir"]).is_dir()
        assert Path(model_payload["sae_best_checkpoint"]).is_file()
        assert Path(model_payload["metrics_path"]).is_file()
        assert Path(model_payload["eval_summary_path"]).is_file()
