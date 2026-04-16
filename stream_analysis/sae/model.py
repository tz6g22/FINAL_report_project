"""TopK sparse autoencoder modules."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SAEConfig
from .utils import resolve_dtype


class TopKActivation(nn.Module):
    """Keep exactly the top-k entries along the latent axis."""

    def __init__(self, k: int) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}.")
        self.k = int(k)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim < 1:
            raise ValueError("TopKActivation expects at least 1D inputs.")
        if self.k > inputs.shape[-1]:
            raise ValueError(
                f"k={self.k} exceeds latent dimension {inputs.shape[-1]}."
            )
        values, indices = torch.topk(inputs, k=self.k, dim=-1)
        sparse = torch.zeros_like(inputs)
        sparse.scatter_(-1, indices, values)
        return sparse


class TopKSAE(nn.Module):
    """TopK SAE with a shared pre-bias on the encoder and decoder path.

    The implemented parameterization is:

    ``z_pre = W_enc (x - b_pre)``
    ``z = TopK(z_pre, k)``
    ``x_hat = W_dec(z) + b_pre``
    """

    def __init__(self, config: SAEConfig) -> None:
        super().__init__()
        self.config = config
        factory_kwargs = {
            "device": None if config.device == "auto" else torch.device(config.device),
            "dtype": resolve_dtype(config.dtype),
        }
        self.b_pre = nn.Parameter(torch.zeros(config.d_in, **factory_kwargs))
        self.W_enc = nn.Parameter(torch.empty(config.n_latents, config.d_in, **factory_kwargs))
        self.W_dec = nn.Parameter(torch.empty(config.n_latents, config.d_in, **factory_kwargs))
        self.activation = TopKActivation(config.k)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize encoder / decoder weights."""

        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5.0))
        if self.config.tied_init:
            with torch.no_grad():
                self.W_dec.copy_(F.normalize(self.W_enc.detach().clone(), dim=1))
        else:
            nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5.0))
        nn.init.zeros_(self.b_pre)
        if self.config.normalize_decoder:
            self.normalize_decoder_weights()

    @property
    def d_in(self) -> int:
        return int(self.config.d_in)

    @property
    def n_latents(self) -> int:
        return int(self.config.n_latents)

    def normalize_decoder_weights(self) -> None:
        """Project decoder atoms back onto the unit sphere."""

        with torch.no_grad():
            normalized = F.normalize(self.W_dec, dim=1, eps=1e-8)
            self.W_dec.copy_(normalized)

    def encode(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode inputs into sparse latent activations."""

        centered = inputs - self.b_pre
        z_pre = F.linear(centered, self.W_enc)
        z = self.activation(z_pre)
        return {"z_pre": z_pre, "z": z}

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode sparse latent activations back to input space."""

        return latents @ self.W_dec + self.b_pre

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        if inputs.ndim != 2:
            raise ValueError(
                f"TopKSAE expects [batch, d_in] inputs, got shape {tuple(inputs.shape)}."
            )
        if inputs.shape[-1] != self.d_in:
            raise ValueError(
                f"Input feature dimension {inputs.shape[-1]} does not match configured d_in={self.d_in}."
            )

        encoded = self.encode(inputs)
        x_hat = self.decode(encoded["z"])
        return {
            "x_hat": x_hat,
            "z": encoded["z"],
            "z_pre": encoded["z_pre"],
            "recon_error": inputs - x_hat,
        }
