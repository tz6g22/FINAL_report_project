"""End-to-end tests for stream_analysis.extract_residuals."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stream_analysis.extract_residuals import main
from scripts.evaluate import load_checkpoint
from toygpt2.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from toygpt2.model import build_model


def _make_analysis_set(root: Path, *, num_samples: int, block_size: int, vocab_size: int) -> Path:
    torch.manual_seed(7)
    input_ids = torch.randint(0, vocab_size, (num_samples, block_size), dtype=torch.long)
    labels = torch.randint(0, vocab_size, (num_samples, block_size), dtype=torch.long)
    target_positions = torch.tensor([0, 3, block_size - 1, 2, 5], dtype=torch.long)[:num_samples]
    payload = {
        "meta": {
            "num_samples": num_samples,
            "block_size": block_size,
            "dataset_type": "tinystories",
        },
        "sample_ids": [f"sample_{idx:02d}" for idx in range(num_samples)],
        "group_labels": ["smoke"] * num_samples,
        "input_ids": input_ids,
        "labels": labels,
        "target_positions": target_positions,
    }
    path = root / "analysis_set.pt"
    torch.save(payload, path)
    return path


def _save_checkpoint(
    root: Path,
    *,
    model_type: str,
    vocab_size: int,
    block_size: int,
    n_layer: int,
    n_head: int,
    n_embd: int,
) -> tuple[Path, Path]:
    experiment = ExperimentConfig(
        model=ModelConfig(
            model_type=model_type,
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=0.0,
        ),
        data=DataConfig(dataset_type="tinystories", block_stride=block_size),
        train=TrainConfig(batch_size=2, max_steps=1, out_dir=str(root)),
    )
    model = build_model(model_type, experiment.model).eval()
    checkpoint = {
        "step": 123,
        "model_type": model_type,
        "model_config": experiment.to_dict()["model"],
        "data_config": experiment.to_dict()["data"],
        "train_config": experiment.to_dict()["train"],
        "model_state": model.state_dict(),
        "optimizer_state": {},
    }

    run_dir = root / model_type
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / f"ckpt_{model_type}_last.pt"
    experiment.save_json(config_path)
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path, config_path


def test_extract_residuals_supports_standard_target_only_and_attnres_both() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        analysis_path = _make_analysis_set(root, num_samples=5, block_size=8, vocab_size=32)
        analysis_set = torch.load(analysis_path, map_location="cpu", weights_only=False)

        cases = [
            ("standard", "target_only"),
            ("attnres", "both"),
        ]
        for model_type, extract_mode in cases:
            checkpoint_path, config_path = _save_checkpoint(
                root,
                model_type=model_type,
                vocab_size=32,
                block_size=8,
                n_layer=4,
                n_head=2,
                n_embd=16,
            )
            output_path = root / model_type / f"{model_type}_{extract_mode}.pt"

            main(
                [
                    "--checkpoint",
                    str(checkpoint_path),
                    "--config",
                    str(config_path),
                    "--analysis-set",
                    str(analysis_path),
                    "--output",
                    str(output_path),
                    "--device",
                    "cpu",
                    "--batch-size",
                    "2",
                    "--extract-mode",
                    extract_mode,
                    "--overwrite",
                ]
            )

            artifact = torch.load(output_path, map_location="cpu", weights_only=False)
            assert artifact["sample_ids"] == analysis_set["sample_ids"]
            assert artifact["group_labels"] == analysis_set["group_labels"]
            assert torch.equal(artifact["target_positions"], analysis_set["target_positions"])
            assert artifact["meta"]["resid_final_definition"] == "post_ln_f_pre_lm_head"

            if extract_mode == "target_only":
                assert "states" in artifact
                assert "states_target" not in artifact
                assert artifact["states"]["resid_0"].shape == (5, 16)
                assert artifact["states"]["resid_final"].shape == (5, 16)
                actual = artifact["states"]["resid_final"]
            else:
                assert "states_target" in artifact
                assert "states_full" in artifact
                assert artifact["states_target"]["resid_1"].shape == (5, 16)
                assert artifact["states_full"]["resid_1"].shape == (5, 8, 16)
                actual = artifact["states_target"]["resid_final"]

            model, experiment, _ = load_checkpoint(checkpoint_path, device=torch.device("cpu"))
            model.eval()
            with torch.no_grad():
                outputs = model(analysis_set["input_ids"], return_intermediates=True)
                cache = outputs["intermediates"]
                final_block = cache[f"blocks.{experiment.model.n_layer - 1}.output"]
                resid_final = model.ln_f(final_block)
                expected = resid_final[
                    torch.arange(analysis_set["input_ids"].shape[0]),
                    analysis_set["target_positions"],
                ].cpu().float()
            assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def _run_all_tests() -> None:
    test_functions = [
        value
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    ]
    for test_fn in sorted(test_functions, key=lambda fn: fn.__name__):
        test_fn()
        print(f"{test_fn.__name__}: ok")
    print(f"{len(test_functions)} tests passed")


if __name__ == "__main__":
    _run_all_tests()
