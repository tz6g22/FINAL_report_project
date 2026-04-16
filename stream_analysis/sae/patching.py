"""SAE-backed activation-override helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from .intervention import (
    encode_with_error,
    invert_preprocessing,
    preprocess_with_context,
    reconstruct_with_feature_edit,
    resolve_feature_ids,
)
from .model import TopKSAE

SUPPORTED_OVERRIDE_SITES = {"input", "attn_out", "mlp_out", "output"}


def site_cache_key(layer_idx: int, site: str) -> str:
    """Return the canonical model cache key for an intervention site."""

    normalized_site = str(site).strip().lower()
    if normalized_site not in SUPPORTED_OVERRIDE_SITES:
        raise ValueError(
            f"Unsupported site {site!r}. Current SAE intervention support is limited to {sorted(SUPPORTED_OVERRIDE_SITES)}."
        )
    if layer_idx < 0:
        raise ValueError("layer_idx must be non-negative.")
    return f"blocks.{layer_idx}.{normalized_site}"


def select_target_rows(site_tensor: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
    """Select one row per sample from a ``[B, T, D]`` site tensor."""

    if site_tensor.ndim != 3:
        raise ValueError(f"site_tensor must have shape [B, T, D], got {tuple(site_tensor.shape)}.")
    if target_positions.ndim != 1 or int(target_positions.shape[0]) != int(site_tensor.shape[0]):
        raise ValueError("target_positions must be a 1D tensor aligned with the site tensor batch.")
    batch_index = torch.arange(site_tensor.shape[0], device=site_tensor.device)
    return site_tensor[batch_index, target_positions]


@dataclass
class SAESiteInterventionOverride:
    """Callable activation override that edits target-position site activations via an SAE."""

    sae: TopKSAE
    target_positions: torch.Tensor
    mode: str
    feature_ids: Sequence[int]
    preprocessing: str = "none"
    scale_factor: float = 1.0
    donor_rows: torch.Tensor | None = None
    preserve_error: bool = True
    last_debug: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.feature_ids = resolve_feature_ids(self.feature_ids, self.sae.n_latents)
        self.target_positions = self.target_positions.detach().cpu().to(torch.long)
        if self.donor_rows is not None:
            if self.donor_rows.ndim != 2 or int(self.donor_rows.shape[1]) != self.sae.d_in:
                raise ValueError(
                    f"donor_rows must have shape [batch, {self.sae.d_in}], got {tuple(self.donor_rows.shape)}."
                )
            self.donor_rows = self.donor_rows.detach().cpu().to(torch.float32)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 3:
            raise ValueError(f"SAE site override expects [batch, seq, d_model], got {tuple(tensor.shape)}.")
        if int(tensor.shape[2]) != self.sae.d_in:
            raise ValueError(
                f"Activation hidden size {tensor.shape[2]} does not match SAE input dim {self.sae.d_in}."
            )
        if int(tensor.shape[0]) != int(self.target_positions.shape[0]):
            raise ValueError(
                "Batch size does not match target_positions. "
                f"Got batch={tensor.shape[0]}, positions={self.target_positions.shape[0]}."
            )

        positions = self.target_positions.to(device=tensor.device)
        batch_index = torch.arange(tensor.shape[0], device=tensor.device)
        raw_rows = tensor[batch_index, positions]
        processed_rows, context = preprocess_with_context(raw_rows, preprocessing=self.preprocessing)

        donor_inputs = None
        donor_debug = None
        if self.donor_rows is not None:
            donor_rows = self.donor_rows.to(device=tensor.device, dtype=raw_rows.dtype)
            donor_inputs, donor_context = preprocess_with_context(donor_rows, preprocessing=self.preprocessing)
            donor_debug = {
                "donor_raw_rows": donor_rows.detach().cpu(),
                "donor_preprocessed_rows": donor_inputs.detach().cpu(),
                "donor_context_mode": donor_context.mode,
            }

        sae_inputs = processed_rows.to(device=next(self.sae.parameters()).device, dtype=self.sae.b_pre.dtype)
        donor_sae_inputs = None
        if donor_inputs is not None:
            donor_sae_inputs = donor_inputs.to(device=sae_inputs.device, dtype=self.sae.b_pre.dtype)

        with torch.no_grad():
            edit_outputs = reconstruct_with_feature_edit(
                self.sae,
                sae_inputs,
                mode=self.mode,
                feature_ids=self.feature_ids,
                scale_factor=self.scale_factor,
                donor_inputs=donor_sae_inputs,
                preserve_error=self.preserve_error,
            )

        rebuilt_processed = edit_outputs["x_tilde"].to(device=tensor.device, dtype=torch.float32)
        rebuilt_raw = invert_preprocessing(rebuilt_processed, context).to(device=tensor.device, dtype=tensor.dtype)
        output = tensor.clone()
        output[batch_index, positions] = rebuilt_raw

        self.last_debug = {
            "target_positions": self.target_positions.detach().cpu(),
            "raw_rows": raw_rows.detach().cpu(),
            "preprocessed_rows": processed_rows.detach().cpu(),
            "rebuilt_processed_rows": rebuilt_processed.detach().cpu(),
            "rebuilt_raw_rows": rebuilt_raw.detach().cpu(),
            "z": edit_outputs["z"].detach().cpu(),
            "edited_z": edit_outputs["edited_z"].detach().cpu(),
            "x_hat": edit_outputs["x_hat"].detach().cpu(),
            "edited_x_hat": edit_outputs["edited_x_hat"].detach().cpu(),
            "err": edit_outputs["err"].detach().cpu(),
            "feature_ids": list(self.feature_ids),
            "mode": self.mode,
            "preprocessing": self.preprocessing,
            "preserve_error": self.preserve_error,
        }
        if donor_debug is not None:
            self.last_debug.update(donor_debug)
        return output


def build_sae_site_override(
    sae: TopKSAE,
    *,
    target_positions: torch.Tensor,
    mode: str,
    feature_ids: Sequence[int],
    preprocessing: str = "none",
    scale_factor: float = 1.0,
    donor_rows: torch.Tensor | None = None,
    preserve_error: bool = True,
) -> SAESiteInterventionOverride:
    """Build a callable override for a target-position SAE feature intervention."""

    return SAESiteInterventionOverride(
        sae=sae,
        target_positions=target_positions,
        mode=mode,
        feature_ids=feature_ids,
        preprocessing=preprocessing,
        scale_factor=scale_factor,
        donor_rows=donor_rows,
        preserve_error=preserve_error,
    )


def extract_rows_from_cache(
    cache: Mapping[str, torch.Tensor],
    *,
    cache_key: str,
    target_positions: torch.Tensor,
) -> torch.Tensor:
    """Extract target-position rows from an intermediates/cache mapping."""

    if cache_key not in cache:
        raise KeyError(f"Missing cache key {cache_key!r}.")
    tensor = cache[cache_key]
    if not torch.is_tensor(tensor):
        raise TypeError(f"cache[{cache_key!r}] must be a torch.Tensor.")
    return select_target_rows(tensor, target_positions.to(device=tensor.device))
