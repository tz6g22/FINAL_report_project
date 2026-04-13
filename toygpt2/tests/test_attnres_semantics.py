"""Assertion-based sanity tests for AttnRes block semantics.

This file is pytest-compatible, but it can also be run directly with:
    python toygpt2/tests/test_attnres_semantics.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from toygpt2.config import ModelConfig
from interp.cache import ActivationCache
from toygpt2.model import build_model


def _make_config(model_type: str, n_layer: int) -> ModelConfig:
    return ModelConfig(
        model_type=model_type,
        vocab_size=32,
        block_size=8,
        n_layer=n_layer,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )


def _make_tokens(config: ModelConfig) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, config.vocab_size, (2, config.block_size))


def _run_model(model_type: str, n_layer: int = 2) -> Tuple[ModelConfig, Dict[str, object]]:
    torch.manual_seed(0)
    config = _make_config(model_type=model_type, n_layer=n_layer)
    model = build_model(model_type, config)
    tokens = _make_tokens(config)
    outputs = model(tokens, return_intermediates=True, return_cache=True)
    return config, outputs


def _expected_history_metadata(n_layer: int) -> List[Dict[str, int | str]]:
    metadata: List[Dict[str, int | str]] = []
    for layer_idx in range(n_layer):
        metadata.append({"site_id": f"blocks.{layer_idx}.pre_attn", "layer_id": layer_idx})
        metadata.append({"site_id": f"blocks.{layer_idx}.pre_mlp", "layer_id": layer_idx})
    return metadata


def test_block_residual_semantics_are_preserved() -> None:
    _, standard_outputs = _run_model("standard", n_layer=1)
    standard_cache = standard_outputs["intermediates"]
    expected_standard_output = (
        standard_cache["blocks.0.input"]
        + standard_cache["blocks.0.attn_out"]
        + standard_cache["blocks.0.mlp_out"]
    )
    assert torch.allclose(standard_cache["blocks.0.output"], expected_standard_output)
    assert not torch.allclose(standard_cache["blocks.0.output"], standard_cache["blocks.0.mlp_out"])

    _, attnres_outputs = _run_model("attnres", n_layer=1)
    attnres_cache = attnres_outputs["intermediates"]
    assert torch.allclose(
        attnres_cache["blocks.0.attnres.pre_attn.current"],
        attnres_cache["blocks.0.input"],
    )
    assert torch.allclose(
        attnres_cache["blocks.0.attnres.pre_mlp.current"],
        attnres_cache["blocks.0.attn_out"],
    )
    assert torch.allclose(attnres_cache["blocks.0.output"], attnres_cache["blocks.0.mlp_out"])


def test_history_grows_by_two_sites_per_layer() -> None:
    for n_layer in (1, 2, 3):
        _, outputs = _run_model("attnres", n_layer=n_layer)
        history = outputs["history"]
        expected_metadata = _expected_history_metadata(n_layer)

        assert len(history) == 2 * n_layer
        assert history.site_ids == [entry["site_id"] for entry in expected_metadata]
        assert history.layer_ids == [entry["layer_id"] for entry in expected_metadata]
        assert outputs["history_metadata"] == expected_metadata
        assert outputs["cache"].metadata["history"] == expected_metadata


def test_attnres_softmax_runs_over_site_dimension() -> None:
    _, outputs = _run_model("attnres", n_layer=3)
    cache = outputs["intermediates"]

    for layer_idx in range(3):
        for site_name in ("pre_attn", "pre_mlp"):
            prefix = f"blocks.{layer_idx}.attnres.{site_name}"
            history_states = cache[f"{prefix}.history_states"]
            scores = cache[f"{prefix}.scores"]
            weights = cache[f"{prefix}.weights"]

            assert scores.shape[2] == history_states.shape[2] + 1
            assert weights.shape == scores.shape
            assert torch.allclose(
                weights.sum(dim=2),
                torch.ones_like(weights.sum(dim=2)),
                atol=1e-5,
                rtol=0.0,
            )


def test_attnres_intermediates_are_shape_aligned() -> None:
    n_layer = 3
    _, outputs = _run_model("attnres", n_layer=n_layer)
    cache = outputs["intermediates"]

    for layer_idx in range(n_layer):
        for site_name in ("pre_attn", "pre_mlp"):
            prefix = f"blocks.{layer_idx}.attnres.{site_name}"
            for suffix in ("current", "history_states", "scores", "weights", "aggregated"):
                assert f"{prefix}.{suffix}" in cache

            current = cache[f"{prefix}.current"]
            history_states = cache[f"{prefix}.history_states"]
            scores = cache[f"{prefix}.scores"]
            weights = cache[f"{prefix}.weights"]
            aggregated = cache[f"{prefix}.aggregated"]

            assert current.ndim == 3
            assert history_states.ndim == 4
            assert scores.ndim == 3
            assert weights.ndim == 3
            assert aggregated.ndim == 3

            assert aggregated.shape == current.shape
            assert history_states.shape[:2] == current.shape[:2]
            assert scores.shape[:2] == current.shape[:2]
            assert weights.shape[:2] == current.shape[:2]
            assert history_states.shape[3] == current.shape[2]


def test_standard_and_attnres_models_share_forward_interface() -> None:
    standard_config, standard_outputs = _run_model("standard", n_layer=2)
    attnres_config, attnres_outputs = _run_model("attnres", n_layer=2)

    expected_shape = (2, standard_config.block_size, standard_config.vocab_size)
    assert standard_outputs["logits"].shape == expected_shape
    assert attnres_outputs["logits"].shape == expected_shape
    assert standard_outputs["logits"].shape == attnres_outputs["logits"].shape
    assert standard_config.block_size == attnres_config.block_size
    assert standard_config.vocab_size == attnres_config.vocab_size

    assert isinstance(standard_outputs["intermediates"], dict)
    assert isinstance(attnres_outputs["intermediates"], dict)
    assert isinstance(standard_outputs["cache"], ActivationCache)
    assert isinstance(attnres_outputs["cache"], ActivationCache)

    assert "blocks.0.output" in standard_outputs["intermediates"]
    assert "blocks.0.attn_out" in standard_outputs["intermediates"]
    assert "blocks.0.attnres.pre_attn.current" in attnres_outputs["intermediates"]
    assert "blocks.0.attnres.pre_mlp.weights" in attnres_outputs["intermediates"]
    assert standard_outputs["cache"].metadata["model_type"] == "standard"
    assert attnres_outputs["cache"].metadata["model_type"] == "attnres"


def _run_all_tests() -> None:
    test_functions: List[Callable[[], None]] = [
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
