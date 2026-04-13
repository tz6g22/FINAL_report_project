"""Model definitions for StandardGPT and AttnResGPT."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attnres import ActivationOverrideMap
from .blocks import AttnResGPTBlock, StandardGPTBlock
from .config import ModelConfig
from .history import ResidualHistory
from interp.cache import ActivationCache


class GPTBase(nn.Module):
    """Common GPT scaffolding shared by both variants."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _embed_tokens(
        self,
        idx: torch.Tensor,
        cache: Dict[str, torch.Tensor] | None = None,
        activation_overrides: ActivationOverrideMap | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError("Sequence length exceeds configured block_size.")

        positions = torch.arange(seq_len, device=idx.device)
        x = self.wte(idx) + self.wpe(positions)[None, :, :]
        x = self.drop(x)

        override = None if activation_overrides is None else activation_overrides.get("embedding_out")
        if override is not None:
            x = override(x) if callable(override) else override
        if cache is not None:
            cache["embedding_out"] = x
        return x

    def _finalize_forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None,
        cache: Dict[str, torch.Tensor] | None,
        return_intermediates: bool,
        return_attn: bool,
        return_cache: bool,
        history: ResidualHistory | None = None,
    ) -> Dict[str, object]:
        x = self.ln_f(x)
        logits = self.lm_head(x)

        result: Dict[str, object] = {"logits": logits}
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            result["loss"] = loss

        if cache is not None:
            if return_intermediates:
                result["intermediates"] = cache
            if return_attn:
                result["attn"] = {
                    key: value
                    for key, value in cache.items()
                    if key.endswith(("attn_probs", "q", "k", "v"))
                }
            if return_cache:
                metadata = {"model_type": self.config.model_type}
                if history is not None:
                    metadata["history"] = history.metadata()
                result["cache"] = ActivationCache(cache, metadata=metadata)
        if history is not None and cache is not None:
            result["history"] = history
            result["history_metadata"] = history.metadata()
        return result


class StandardGPT(GPTBase):
    """Standard GPT-2 style baseline with residual adds."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.blocks = nn.ModuleList([StandardGPTBlock(config, layer_idx=i) for i in range(config.n_layer)])

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_intermediates: bool = False,
        return_attn: bool = False,
        return_cache: bool = False,
        activation_overrides: ActivationOverrideMap | None = None,
    ) -> Dict[str, object]:
        use_cache = return_intermediates or return_attn or return_cache or activation_overrides is not None
        cache: Dict[str, torch.Tensor] | None = {} if use_cache else None

        x = self._embed_tokens(idx, cache=cache, activation_overrides=activation_overrides)
        for block in self.blocks:
            x = block(x, cache=cache, activation_overrides=activation_overrides)

        return self._finalize_forward(
            x=x,
            targets=targets,
            cache=cache,
            return_intermediates=return_intermediates,
            return_attn=return_attn,
            return_cache=return_cache,
        )


class AttnResGPT(GPTBase):
    """Decoder-only GPT variant with Kimi-style Attention Residuals."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.blocks = nn.ModuleList([AttnResGPTBlock(config, layer_idx=i) for i in range(config.n_layer)])

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_intermediates: bool = False,
        return_attn: bool = False,
        return_cache: bool = False,
        activation_overrides: ActivationOverrideMap | None = None,
    ) -> Dict[str, object]:
        use_cache = return_intermediates or return_attn or return_cache or activation_overrides is not None
        cache: Dict[str, torch.Tensor] | None = {} if use_cache else None

        x = self._embed_tokens(idx, cache=cache, activation_overrides=activation_overrides)
        history = ResidualHistory()
        for block in self.blocks:
            x, history = block(x, history=history, cache=cache, activation_overrides=activation_overrides)

        return self._finalize_forward(
            x=x,
            targets=targets,
            cache=cache,
            return_intermediates=return_intermediates,
            return_attn=return_attn,
            return_cache=return_cache,
            history=history,
        )


def build_model(model_type: str, config: ModelConfig) -> GPTBase:
    """Factory used by training, evaluation, and demos."""

    config.model_type = model_type
    if model_type == "standard":
        return StandardGPT(config)
    if model_type == "attnres":
        return AttnResGPT(config)
    raise ValueError(f"Unknown model_type: {model_type}")
