"""Unit tests for MemorizationPatchingRunner.

This file is pytest-compatible and can also run directly:
    python toygpt2/tests/test_memorization_runner.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from toygpt2.config import ModelConfig
from toygpt2.interp.memorization_runner import MemorizationPatchingRunner
from toygpt2.model import build_model


def _make_model(model_type: str) -> torch.nn.Module:
    config = ModelConfig(
        model_type=model_type,
        vocab_size=32,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )
    return build_model(model_type, config)


def _make_pair() -> tuple[torch.Tensor, torch.Tensor]:
    clean = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    corrupted = clean.clone()
    corrupted[0, 2] = 9
    return clean, corrupted


def test_runner_direct_patching_and_position_shapes() -> None:
    model = _make_model("standard")
    clean, corrupted = _make_pair()
    runner = MemorizationPatchingRunner(model)

    result = runner.run(
        clean_input=clean,
        corrupted_input=corrupted,
        patch_site="blocks.0.attn_out",
        target_position=7,
        target_token_id=int(clean[0, 7].item()),
        metric="logprob",
    )

    assert result.model_type == "standard"
    assert result.patch_site == "blocks.0.attn_out"
    assert result.baseline_clean_score.shape == (1,)
    assert result.baseline_corrupted_score.shape == (1,)
    assert result.patched_score.shape == (1,)
    assert result.effect_size.shape == (1,)
    assert result.position_scores["clean"].shape == (1, 8)
    assert result.position_scores["corrupted"].shape == (1, 8)
    assert result.position_scores["patched"].shape == (1, 8)
    assert result.position_scores["effect"].shape == (1, 8)


def test_runner_site_sweep_produces_site_axis() -> None:
    model = _make_model("standard")
    clean, corrupted = _make_pair()
    runner = MemorizationPatchingRunner(model)

    sweep = runner.run_site_sweep(
        clean_input=clean,
        corrupted_input=corrupted,
        patch_sites=["blocks.0.attn_out", "blocks.1.mlp_out"],
        target_position=7,
        target_token_id=int(clean[0, 7].item()),
        metric="logit",
    )

    assert sweep.effect_by_site.shape == (2, 1)
    assert sweep.position_effect_by_site.shape == (2, 1, 8)
    assert sweep.patch_sites == ["blocks.0.attn_out", "blocks.1.mlp_out"]


def test_runner_supports_attnres_without_forcing_standard_semantics() -> None:
    model = _make_model("attnres")
    clean, corrupted = _make_pair()
    runner = MemorizationPatchingRunner(model)

    result = runner.run(
        clean_input=clean,
        corrupted_input=corrupted,
        patch_site="blocks.0.attnres.pre_attn.aggregated",
        target_position=7,
        target_token_id=int(clean[0, 7].item()),
        metric="prob",
    )

    assert result.model_type == "attnres"
    assert result.clean_trace is not None
    assert result.clean_trace.layer(0).block_type == "attnres"
    assert result.clean_trace.layer(0).residual_after_attn is None


def test_runner_errors_on_missing_patch_site() -> None:
    model = _make_model("standard")
    clean, corrupted = _make_pair()
    runner = MemorizationPatchingRunner(model)

    try:
        runner.run(
            clean_input=clean,
            corrupted_input=corrupted,
            patch_site="blocks.999.attn_out",
            target_position=7,
            target_token_id=int(clean[0, 7].item()),
        )
        raise AssertionError("Expected KeyError for missing patch site.")
    except KeyError as error:
        assert "missing from clean cache" in str(error)


def test_runner_errors_on_target_position_out_of_range() -> None:
    model = _make_model("standard")
    clean, corrupted = _make_pair()
    runner = MemorizationPatchingRunner(model)

    try:
        runner.run(
            clean_input=clean,
            corrupted_input=corrupted,
            patch_site="blocks.0.attn_out",
            target_position=8,
            target_token_id=int(clean[0, 7].item()),
        )
        raise AssertionError("Expected IndexError for out-of-range target_position.")
    except IndexError as error:
        assert "out of range" in str(error)


def test_runner_errors_on_shape_mismatch() -> None:
    model = _make_model("standard")
    clean, _ = _make_pair()
    corrupted = torch.tensor([[1, 2, 3]], dtype=torch.long)
    runner = MemorizationPatchingRunner(model)

    try:
        runner.run(
            clean_input=clean,
            corrupted_input=corrupted,
            patch_site="blocks.0.attn_out",
            target_position=2,
            target_token_id=3,
        )
        raise AssertionError("Expected ValueError for shape mismatch.")
    except ValueError as error:
        assert "shapes must match" in str(error)


def test_runner_errors_on_invalid_metric() -> None:
    model = _make_model("standard")
    clean, corrupted = _make_pair()
    runner = MemorizationPatchingRunner(model)

    try:
        runner.run(
            clean_input=clean,
            corrupted_input=corrupted,
            patch_site="blocks.0.attn_out",
            target_position=7,
            target_token_id=int(clean[0, 7].item()),
            metric="invalid_metric",
        )
        raise AssertionError("Expected ValueError for invalid metric.")
    except ValueError as error:
        assert "Unsupported metric" in str(error)


def test_runner_errors_on_target_token_out_of_vocab_range() -> None:
    model = _make_model("standard")
    clean, corrupted = _make_pair()
    runner = MemorizationPatchingRunner(model)

    try:
        runner.run(
            clean_input=clean,
            corrupted_input=corrupted,
            patch_site="blocks.0.attn_out",
            target_position=7,
            target_token_id=10_000,
            metric="logit",
        )
        raise AssertionError("Expected IndexError for out-of-range target_token_id.")
    except IndexError as error:
        assert "out of range for vocab_size" in str(error)


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
