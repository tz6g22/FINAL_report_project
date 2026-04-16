"""Loss helpers for TopK SAE training and evaluation."""

from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn.functional as F


def reconstruction_mse(inputs: torch.Tensor, reconstructions: torch.Tensor) -> torch.Tensor:
    """Mean-squared reconstruction error."""

    return F.mse_loss(reconstructions, inputs)


def avg_l0(latents: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Average number of active latents per sample."""

    active = (latents.abs() > threshold).float()
    return active.sum(dim=-1).mean()


def dead_latent_stats(latents: torch.Tensor, threshold: float = 0.0) -> Dict[str, float]:
    """Return simple dead-latent diagnostics for one batch."""

    if latents.ndim != 2:
        raise ValueError(f"latents must have shape [batch, n_latents], got {tuple(latents.shape)}.")
    active = latents.abs() > threshold
    fired = active.any(dim=0)
    dead = ~fired
    density = float(active.float().mean().item())
    return {
        "dead_latent_count": float(dead.sum().item()),
        "alive_latent_count": float(fired.sum().item()),
        "dead_latent_frac": float(dead.float().mean().item()),
        "activation_density": density,
    }


def auxk_loss(z_pre: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Optional auxiliary loss that suppresses non-selected preactivations.

    This regularizer keeps the inactive tail of ``z_pre`` small, which makes the
    TopK routing more decisive without changing the main objective away from
    reconstruction.
    """

    inactive_mask = z == 0
    inactive_values = z_pre.masked_fill(~inactive_mask, 0.0)
    return inactive_values.pow(2).mean()


def compute_loss_dict(
    inputs: torch.Tensor,
    outputs: Mapping[str, torch.Tensor],
    *,
    use_auxk: bool,
    auxk_alpha: float,
    dead_threshold: float = 0.0,
) -> Dict[str, object]:
    """Compute the unified SAE loss dictionary used by training and eval."""

    required = {"x_hat", "z", "z_pre"}
    missing = required - set(outputs.keys())
    if missing:
        raise KeyError(f"SAE outputs are missing required keys: {sorted(missing)}.")

    recon = reconstruction_mse(inputs, outputs["x_hat"])
    aux = auxk_loss(outputs["z_pre"], outputs["z"]) if use_auxk else recon.new_tensor(0.0)
    total = recon + auxk_alpha * aux
    stats = dead_latent_stats(outputs["z"], threshold=dead_threshold)
    l0 = avg_l0(outputs["z"], threshold=dead_threshold)

    return {
        "loss": total,
        "recon_mse": recon,
        "auxk_loss": aux,
        "avg_l0": l0,
        **stats,
    }
