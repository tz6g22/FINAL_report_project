"""Core SAE feature-edit helpers used by intervention experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from .config import resolve_preprocessing_mode
from .model import TopKSAE

SUPPORTED_EDIT_MODES = {"zero", "keep_only", "scale", "patch"}


@dataclass
class PreprocessingContext:
    """Sufficient statistics required to invert SAE input preprocessing."""

    mode: str
    mean: torch.Tensor | None = None
    norm: torch.Tensor | None = None


def resolve_feature_ids(feature_ids: Iterable[int] | None, n_latents: int) -> list[int]:
    """Validate and normalize a feature-id collection."""

    if feature_ids is None:
        return []

    normalized: list[int] = []
    for raw_feature_id in feature_ids:
        feature_id = int(raw_feature_id)
        if feature_id < 0 or feature_id >= n_latents:
            raise IndexError(f"feature_id {feature_id} is out of range [0, {n_latents - 1}].")
        normalized.append(feature_id)
    if not normalized:
        return []
    return sorted(set(normalized))


def feature_ids_to_mask(
    feature_ids: Iterable[int],
    *,
    n_latents: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Convert feature ids into a boolean mask."""

    feature_list = resolve_feature_ids(feature_ids, n_latents)
    mask = torch.zeros(n_latents, dtype=torch.bool, device=device)
    if feature_list:
        mask[torch.tensor(feature_list, dtype=torch.long, device=device)] = True
    return mask


def preprocess_with_context(
    inputs: torch.Tensor,
    *,
    preprocessing: str = "none",
) -> tuple[torch.Tensor, PreprocessingContext]:
    """Apply SAE preprocessing while retaining the information needed to invert it."""

    mode = resolve_preprocessing_mode(preprocessing)
    outputs = inputs.to(dtype=torch.float32)
    mean = None
    norm = None

    if "mean-center" in mode:
        mean = outputs.mean(dim=-1, keepdim=True)
        outputs = outputs - mean
    if "unit-norm" in mode:
        norm = outputs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        outputs = outputs / norm

    return outputs, PreprocessingContext(mode=mode, mean=mean, norm=norm)


def invert_preprocessing(inputs: torch.Tensor, context: PreprocessingContext) -> torch.Tensor:
    """Map a processed SAE-space tensor back to the raw model-activation space."""

    outputs = inputs
    if "unit-norm" in context.mode:
        if context.norm is None:
            raise ValueError("Preprocessing context is missing norm statistics for unit-norm inversion.")
        outputs = outputs * context.norm
    if "mean-center" in context.mode:
        if context.mean is None:
            raise ValueError("Preprocessing context is missing mean statistics for mean-center inversion.")
        outputs = outputs + context.mean
    return outputs


def encode_with_error(sae: TopKSAE, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Encode one SAE-space activation batch and expose its reconstruction error."""

    outputs = sae(inputs)
    return {
        "z": outputs["z"],
        "z_pre": outputs["z_pre"],
        "x_hat": outputs["x_hat"],
        "err": outputs["recon_error"],
    }


def edit_feature_activations(
    latents: torch.Tensor,
    *,
    mode: str,
    feature_ids: Sequence[int],
    scale_factor: float = 1.0,
    donor_latents: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a feature edit to a latent batch."""

    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in SUPPORTED_EDIT_MODES:
        raise ValueError(f"Unsupported edit mode {mode!r}. Expected one of {sorted(SUPPORTED_EDIT_MODES)}.")
    if latents.ndim != 2:
        raise ValueError(f"latents must have shape [batch, n_latents], got {tuple(latents.shape)}.")

    feature_list = resolve_feature_ids(feature_ids, int(latents.shape[1]))
    if normalized_mode in {"zero", "scale", "patch"} and not feature_list:
        raise ValueError(f"mode={normalized_mode!r} requires at least one feature id.")
    if normalized_mode == "keep_only" and not feature_list:
        raise ValueError("keep_only requires a non-empty feature set.")

    edited = latents.clone()
    if normalized_mode == "zero":
        edited[:, feature_list] = 0.0
        return edited
    if normalized_mode == "keep_only":
        keep_mask = feature_ids_to_mask(feature_list, n_latents=int(latents.shape[1]), device=latents.device)
        return edited * keep_mask.to(dtype=edited.dtype).unsqueeze(0)
    if normalized_mode == "scale":
        edited[:, feature_list] = edited[:, feature_list] * float(scale_factor)
        return edited

    if donor_latents is None:
        raise ValueError("patch mode requires donor_latents.")
    if donor_latents.shape != latents.shape:
        raise ValueError(
            "donor_latents must match latents shape. "
            f"Got donor={tuple(donor_latents.shape)}, latents={tuple(latents.shape)}."
        )
    donor_latents = donor_latents.to(device=latents.device, dtype=latents.dtype)
    edited[:, feature_list] = donor_latents[:, feature_list]
    return edited


def rebuild_activation(
    edited_x_hat: torch.Tensor,
    err: torch.Tensor,
    *,
    preserve_error: bool = True,
) -> torch.Tensor:
    """Rebuild an activation after editing the SAE reconstruction."""

    if edited_x_hat.shape != err.shape:
        raise ValueError(
            f"edited_x_hat and err must have matching shapes, got {tuple(edited_x_hat.shape)} vs {tuple(err.shape)}."
        )
    return edited_x_hat + err if preserve_error else edited_x_hat


def reconstruct_with_feature_edit(
    sae: TopKSAE,
    inputs: torch.Tensor,
    *,
    mode: str,
    feature_ids: Sequence[int],
    scale_factor: float = 1.0,
    donor_inputs: torch.Tensor | None = None,
    donor_latents: torch.Tensor | None = None,
    preserve_error: bool = True,
) -> dict[str, torch.Tensor]:
    """Encode, edit, decode, and rebuild one SAE-space activation batch."""

    encoded = encode_with_error(sae, inputs)

    resolved_donor_latents = donor_latents
    if str(mode).strip().lower() == "patch" and resolved_donor_latents is None:
        if donor_inputs is None:
            raise ValueError("patch mode requires donor_inputs or donor_latents.")
        resolved_donor_latents = encode_with_error(sae, donor_inputs)["z"]

    edited_z = edit_feature_activations(
        encoded["z"],
        mode=mode,
        feature_ids=feature_ids,
        scale_factor=scale_factor,
        donor_latents=resolved_donor_latents,
    )
    edited_x_hat = sae.decode(edited_z)
    x_tilde = rebuild_activation(edited_x_hat, encoded["err"], preserve_error=preserve_error)
    return {
        "z": encoded["z"],
        "z_pre": encoded["z_pre"],
        "x_hat": encoded["x_hat"],
        "err": encoded["err"],
        "edited_z": edited_z,
        "edited_x_hat": edited_x_hat,
        "x_tilde": x_tilde,
    }
