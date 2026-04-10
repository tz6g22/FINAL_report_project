"""Unit tests for unified analysis adapter.

This file is pytest-compatible and can also run directly:
    python mini_gpt_attnres/tests/test_analysis_adapter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_gpt_attnres.config import ModelConfig
from mini_gpt_attnres.interp.analysis_adapter import AnalysisAdapter
from mini_gpt_attnres.model import build_model


def _make_tokens(vocab_size: int, block_size: int) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, vocab_size, (1, block_size))


def _run_outputs(model_type: str, n_layer: int = 2) -> dict[str, object]:
    config = ModelConfig(
        model_type=model_type,
        vocab_size=32,
        block_size=8,
        n_layer=n_layer,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )
    model = build_model(model_type, config)
    tokens = _make_tokens(config.vocab_size, config.block_size)
    with torch.no_grad():
        return model(tokens, return_intermediates=True, return_cache=True)


def test_analysis_adapter_maps_standard_fields() -> None:
    outputs = _run_outputs("standard", n_layer=2)
    trace = AnalysisAdapter.from_model_outputs(outputs)

    assert trace.model_type == "standard"
    assert len(trace.records) == 2

    record = trace.layer(0)
    assert record.block_type == "standard"
    assert record.layer_input is not None
    assert record.attn_out is not None
    assert record.mlp_out is not None
    assert record.layer_output is not None
    assert record.residual_after_attn is not None
    assert record.attnres_pre_attn is None
    assert record.attnres_pre_mlp is None
    assert "attnres_pre_attn" in record.unavailable
    assert "attnres_pre_mlp" in record.unavailable


def test_analysis_adapter_maps_attnres_fields() -> None:
    outputs = _run_outputs("attnres", n_layer=2)
    trace = AnalysisAdapter.from_model_outputs(outputs)

    assert trace.model_type == "attnres"
    assert len(trace.records) == 2

    record = trace.layer(0)
    assert record.block_type == "attnres"
    assert record.attnres_pre_attn is not None
    assert record.attnres_pre_mlp is not None
    assert record.attnres_pre_attn.current is not None
    assert record.attnres_pre_attn.history_states is not None
    assert record.attnres_pre_mlp.weights is not None
    assert record.residual_after_attn is None
    assert "residual_after_attn" in record.unavailable


def test_analysis_adapter_reports_unavailable_fields_explicitly() -> None:
    outputs = _run_outputs("standard", n_layer=1)
    cache = dict(outputs["intermediates"])
    cache.pop("blocks.0.mlp_out")

    trace = AnalysisAdapter.from_cache_dict(cache, model_type="standard")
    record = trace.layer(0)
    assert record.mlp_out is None
    assert "mlp_out" in record.unavailable
    assert "missing cache key" in record.unavailable["mlp_out"]


def test_analysis_adapter_handles_missing_attnres_site_fields_gracefully() -> None:
    outputs = _run_outputs("attnres", n_layer=1)
    cache = dict(outputs["intermediates"])
    cache.pop("blocks.0.attnres.pre_attn.weights")

    trace = AnalysisAdapter.from_cache_dict(cache, model_type="attnres")
    record = trace.layer(0)
    assert record.attnres_pre_attn is not None
    assert record.attnres_pre_attn.weights is None
    assert "weights" in record.attnres_pre_attn.unavailable


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
