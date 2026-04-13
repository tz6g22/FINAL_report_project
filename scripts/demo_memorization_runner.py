"""Minimal demo for AnalysisAdapter + MemorizationPatchingRunner."""

from __future__ import annotations

import torch

from toygpt2.config import ModelConfig
from interp import AnalysisAdapter, MemorizationPatchingRunner
from toygpt2.model import build_model


def main() -> None:
    torch.manual_seed(0)
    config = ModelConfig(
        model_type="standard",
        vocab_size=32,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )
    model = build_model("standard", config)

    clean = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    corrupted = clean.clone()
    corrupted[0, 2] = 9

    runner = MemorizationPatchingRunner(model)
    result = runner.run(
        clean_input=clean,
        corrupted_input=corrupted,
        patch_site="blocks.0.attn_out",
        target_position=7,
        target_token_id=int(clean[0, 7].item()),
        metric="logprob",
    )

    print("baseline_clean_score:", result.baseline_clean_score.tolist())
    print("baseline_corrupted_score:", result.baseline_corrupted_score.tolist())
    print("patched_score:", result.patched_score.tolist())
    print("effect_size:", result.effect_size.tolist())
    print("position_effect_shape:", tuple(result.position_scores["effect"].shape))

    outputs = model(clean, return_intermediates=True, return_cache=True)
    trace = AnalysisAdapter.from_model_outputs(outputs)
    print("trace layers:", list(trace.layers()))
    print("layer0 block_type:", trace.layer(0).block_type)


if __name__ == "__main__":
    main()

