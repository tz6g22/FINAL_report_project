"""Small forward-pass demo for StandardGPT and AttnResGPT."""

from __future__ import annotations

import torch

from toygpt2.config import ModelConfig
from toygpt2.model import build_model


def main() -> None:
    torch.manual_seed(0)
    base_config = ModelConfig(
        vocab_size=32,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
    )
    tokens = torch.randint(0, base_config.vocab_size, (2, base_config.block_size))

    for model_type in ["standard", "attnres"]:
        config = ModelConfig(**{**base_config.__dict__, "model_type": model_type})
        model = build_model(model_type, config)
        outputs = model(tokens, return_intermediates=True, return_attn=True, return_cache=True)

        print(f"{model_type} logits shape: {tuple(outputs['logits'].shape)}")
        print(f"{model_type} intermediate keys:")
        for key in sorted(outputs["intermediates"].keys()):
            print(f"  {key}")

        if model_type == "attnres":
            history = outputs["history"]
            expected_length = config.n_layer * 2
            print(f"AttnRes history length: {len(history)} (expected {expected_length})")
            if len(history) != expected_length:
                raise RuntimeError("AttnRes history length did not grow by two sites per layer.")


if __name__ == "__main__":
    main()
