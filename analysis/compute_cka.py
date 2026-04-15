#!/usr/bin/env python3
"""Compute linear CKA from activation artifacts exported by extract_residuals.py.

This script reads one or two activation artifact ``.pt`` files and computes
linear CKA (Centered Kernel Alignment) over residual-stream states. It is meant
for downstream analysis only: it does not extract activations, train probes, or
modify training/model code.

Supported artifact layouts
--------------------------
The main path targets ``target_only`` artifacts whose state tensors have shape
``[N, D]`` and are stored under:

- ``states`` for ``extract_mode=target_only``
- ``states_target`` for ``extract_mode=both``

If an artifact only contains full-sequence states (for example
``states``/``states_full`` with shape ``[N, T, D]``), the script raises a clear
error and asks for a target-only artifact.

Linear CKA definition
---------------------
This script implements **linear CKA** for two representation matrices
``X in R^{N x D_x}`` and ``Y in R^{N x D_y}``, where rows are aligned samples.

- ``feature`` centering computes:

  ``CKA(X, Y) = ||X_c^T Y_c||_F^2 / (||X_c^T X_c||_F * ||Y_c^T Y_c||_F)``

  where ``X_c`` and ``Y_c`` are feature-centered across the sample dimension.

- ``gram`` centering computes the equivalent linear-kernel form by building
  ``K = X X^T`` and ``L = Y Y^T``, centering the Gram matrices across samples,
  then normalizing their Frobenius inner product.

This is well suited for comparing residual stream / block-input states because
it measures representational similarity between two sample-aligned activation
matrices while being invariant to isotropic rescaling.

Usage examples
--------------
::

    python analysis/compute_cka.py \\
      --artifact-a artifacts/activations/standard_last_target.pt \\
      --mode within \\
      --output artifacts/cka/standard_last_within.pt

    python analysis/compute_cka.py \\
      --artifact-a artifacts/activations/standard_last_target.pt \\
      --artifact-b artifacts/activations/attnres_last_target.pt \\
      --mode cross_same_layer \\
      --output artifacts/cka/standard_vs_attnres_same_layer.pt

    python analysis/compute_cka.py \\
      --artifact-a artifacts/activations/standard_step1000_target.pt \\
      --artifact-b artifacts/activations/standard_step2000_target.pt \\
      --mode cross_same_layer \\
      --output artifacts/cka/standard_step1000_vs_step2000.pt
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

logger = logging.getLogger("compute_cka")

_SCRIPT_VERSION = "1.0.1"
_TARGET_STATE_COLLECTION = "states"
_BOTH_TARGET_COLLECTION = "states_target"
_BOTH_FULL_COLLECTION = "states_full"


@dataclass
class ActivationArtifact:
    """In-memory view of an activation artifact used for CKA.

    Parameters
    ----------
    path:
        Resolved artifact path for provenance and logging.
    meta:
        Top-level artifact metadata mapping.
    sample_ids:
        Ordered sample identifiers. These define the alignment axis for CKA.
    group_labels:
        Ordered group labels aligned with ``sample_ids``.
    target_positions:
        Tensor of shape ``[N]`` aligned with ``sample_ids``.
    states:
        Residual-state mapping. The main path expects tensors of shape ``[N, D]``.
    state_collection_name:
        Which top-level field supplied ``states`` (for example ``states`` or
        ``states_target``).
    """

    path: Path
    meta: Dict[str, Any]
    sample_ids: List[str]
    group_labels: List[str]
    target_positions: torch.Tensor
    states: Dict[str, torch.Tensor]
    state_collection_name: str

    @property
    def num_samples(self) -> int:
        return len(self.sample_ids)


def _state_sort_key(name: str) -> tuple[int, str]:
    """Sort ``resid_0 ... resid_{L-1}, resid_final`` stably before unknown names."""

    if name == "resid_final":
        return (10**9, name)
    if name.startswith("resid_"):
        suffix = name.split("_", maxsplit=1)[1]
        try:
            return (int(suffix), name)
        except ValueError:
            return (10**9 + 1, name)
    return (10**9 + 2, name)


def sort_state_names(names: Iterable[str]) -> List[str]:
    """Return state names in the project's expected residual-stream order."""

    return sorted(names, key=_state_sort_key)


def _validate_string_sequence(
    values: object,
    *,
    name: str,
    num_samples: int,
    require_unique: bool = False,
) -> List[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings.")
    output = list(values)
    if len(output) != num_samples:
        raise ValueError(f"{name} length {len(output)} != num_samples {num_samples}.")
    for index, value in enumerate(output):
        if not isinstance(value, str):
            raise TypeError(f"{name}[{index}] must be a string, got {type(value).__name__}.")
    if require_unique and len(set(output)) != len(output):
        raise ValueError(f"{name} contains duplicates.")
    return output


def _clone_state_mapping(states: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    if not isinstance(states, Mapping):
        raise TypeError(f"State collection must be a mapping, got {type(states).__name__}.")
    if not states:
        raise ValueError("State collection is empty.")

    output: Dict[str, torch.Tensor] = {}
    for state_name, tensor in states.items():
        if not isinstance(state_name, str):
            raise TypeError(f"State name must be a string, got {type(state_name).__name__}.")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"State {state_name!r} must be a torch.Tensor, got {type(tensor).__name__}."
            )
        output[state_name] = tensor.detach().cpu().clone()
    return output


def get_state_dict(payload: Mapping[str, Any]) -> tuple[Dict[str, torch.Tensor], str]:
    """Select the target-position state collection from an artifact payload."""

    if _TARGET_STATE_COLLECTION in payload:
        return _clone_state_mapping(payload[_TARGET_STATE_COLLECTION]), _TARGET_STATE_COLLECTION
    if _BOTH_TARGET_COLLECTION in payload:
        return _clone_state_mapping(payload[_BOTH_TARGET_COLLECTION]), _BOTH_TARGET_COLLECTION
    if _BOTH_FULL_COLLECTION in payload:
        raise ValueError(
            "Artifact contains 'states_full' but no target-position state collection. "
            "This script expects target-only activations with shape [N, D]."
        )
    raise ValueError(
        "Artifact is missing a usable state collection. Expected one of "
        f"'{_TARGET_STATE_COLLECTION}' or '{_BOTH_TARGET_COLLECTION}'."
    )


def load_artifact(path: str | Path) -> ActivationArtifact:
    """Load and validate one activation artifact."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Artifact not found: {resolved}")

    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Artifact must be a mapping, got {type(payload).__name__}.")

    meta = payload.get("meta")
    if not isinstance(meta, Mapping):
        raise TypeError("Artifact must contain a 'meta' mapping.")

    states, state_collection_name = get_state_dict(payload)
    if not states:
        raise ValueError("Artifact does not contain any states.")

    num_samples = int(next(iter(states.values())).shape[0])
    sample_ids = _validate_string_sequence(
        payload.get("sample_ids"),
        name="sample_ids",
        num_samples=num_samples,
        require_unique=True,
    )
    group_labels = _validate_string_sequence(
        payload.get("group_labels"),
        name="group_labels",
        num_samples=num_samples,
    )

    target_positions = payload.get("target_positions")
    if not isinstance(target_positions, torch.Tensor):
        raise TypeError("Artifact must contain 'target_positions' as a torch.Tensor.")
    if target_positions.ndim != 1 or int(target_positions.shape[0]) != num_samples:
        raise ValueError(
            "target_positions must have shape [N]. "
            f"Got {tuple(target_positions.shape)} for num_samples={num_samples}."
        )
    if torch.is_floating_point(target_positions):
        raise TypeError("target_positions must contain integer indices.")

    return ActivationArtifact(
        path=resolved,
        meta=dict(meta),
        sample_ids=sample_ids,
        group_labels=group_labels,
        target_positions=target_positions.to(torch.long).cpu().clone(),
        states=states,
        state_collection_name=state_collection_name,
    )


def _index_artifact(artifact: ActivationArtifact, indices: torch.Tensor) -> ActivationArtifact:
    """Return a reordered/subsetted artifact along the sample axis."""

    if indices.ndim != 1:
        raise ValueError(f"indices must be 1D, got shape {tuple(indices.shape)}.")
    indices = indices.to(torch.long).cpu()
    return ActivationArtifact(
        path=artifact.path,
        meta=dict(artifact.meta),
        sample_ids=[artifact.sample_ids[int(i)] for i in indices.tolist()],
        group_labels=[artifact.group_labels[int(i)] for i in indices.tolist()],
        target_positions=artifact.target_positions.index_select(0, indices),
        states={name: tensor.index_select(0, indices) for name, tensor in artifact.states.items()},
        state_collection_name=artifact.state_collection_name,
    )


def align_artifacts_by_sample_ids(
    artifact_a: ActivationArtifact,
    artifact_b: ActivationArtifact,
) -> tuple[ActivationArtifact, ActivationArtifact, bool]:
    """Align two artifacts by ``sample_ids``.

    The default contract is strict:
    - if ``sample_ids`` already match exactly, no reordering is done
    - if the sets match but order differs, artifact B is reordered to match A
    - if the sets differ, the function raises an error

    After alignment, ``group_labels`` and ``target_positions`` must also match.
    """

    if artifact_a.sample_ids == artifact_b.sample_ids:
        reordered = False
    else:
        ids_a = set(artifact_a.sample_ids)
        ids_b = set(artifact_b.sample_ids)
        if ids_a != ids_b:
            missing_from_b = sorted(ids_a - ids_b)
            missing_from_a = sorted(ids_b - ids_a)
            raise ValueError(
                "Artifacts do not contain the same sample_ids. "
                f"Missing from artifact_b (first 5): {missing_from_b[:5]}; "
                f"missing from artifact_a (first 5): {missing_from_a[:5]}."
            )
        index_b = {sample_id: idx for idx, sample_id in enumerate(artifact_b.sample_ids)}
        reorder_indices = torch.tensor(
            [index_b[sample_id] for sample_id in artifact_a.sample_ids],
            dtype=torch.long,
        )
        artifact_b = _index_artifact(artifact_b, reorder_indices)
        reordered = True
        logger.info("reordered artifact_b to match sample_ids from artifact_a")

    if artifact_a.group_labels != artifact_b.group_labels:
        for idx, (label_a, label_b) in enumerate(zip(artifact_a.group_labels, artifact_b.group_labels)):
            if label_a != label_b:
                raise ValueError(
                    "group_labels mismatch after sample alignment at index "
                    f"{idx} for sample_id={artifact_a.sample_ids[idx]!r}: "
                    f"{label_a!r} != {label_b!r}."
                )
        raise ValueError("group_labels mismatch after sample alignment.")

    if not torch.equal(artifact_a.target_positions, artifact_b.target_positions):
        mismatch = torch.nonzero(artifact_a.target_positions != artifact_b.target_positions, as_tuple=False)
        first = int(mismatch[0].item()) if int(mismatch.numel()) > 0 else -1
        raise ValueError(
            "target_positions mismatch after sample alignment. "
            f"First mismatch index={first}, sample_id={artifact_a.sample_ids[first]!r}."
        )

    return artifact_a, artifact_b, reordered


def apply_group_filter(
    artifact_a: ActivationArtifact,
    artifact_b: Optional[ActivationArtifact],
    group_filter: Optional[str],
) -> tuple[ActivationArtifact, Optional[ActivationArtifact]]:
    """Filter samples by a single group label."""

    if group_filter is None:
        return artifact_a, artifact_b

    selected = [idx for idx, label in enumerate(artifact_a.group_labels) if label == group_filter]
    if not selected:
        available = sorted(set(artifact_a.group_labels))
        raise ValueError(
            f"group_filter={group_filter!r} matched 0 samples. "
            f"Available group labels: {available}"
        )

    indices = torch.tensor(selected, dtype=torch.long)
    artifact_a = _index_artifact(artifact_a, indices)
    artifact_b = None if artifact_b is None else _index_artifact(artifact_b, indices)
    return artifact_a, artifact_b


def apply_sample_limit(
    artifact_a: ActivationArtifact,
    artifact_b: Optional[ActivationArtifact],
    sample_limit: Optional[int],
) -> tuple[ActivationArtifact, Optional[ActivationArtifact]]:
    """Keep only the first ``sample_limit`` aligned samples when requested."""

    if sample_limit is None or sample_limit >= artifact_a.num_samples:
        return artifact_a, artifact_b

    indices = torch.arange(sample_limit, dtype=torch.long)
    artifact_a = _index_artifact(artifact_a, indices)
    artifact_b = None if artifact_b is None else _index_artifact(artifact_b, indices)
    return artifact_a, artifact_b


def _validate_selected_state_tensor(
    tensor: torch.Tensor,
    *,
    state_name: str,
    artifact_label: str,
    num_samples: int,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{artifact_label}[{state_name!r}] must be a torch.Tensor.")
    if not tensor.is_floating_point():
        raise TypeError(
            f"{artifact_label}[{state_name!r}] must be floating point, got dtype={tensor.dtype}."
        )
    if tensor.ndim != 2:
        if tensor.ndim == 3:
            raise ValueError(
                f"{artifact_label}[{state_name!r}] has shape {tuple(tensor.shape)}. "
                "This looks like a full-sequence activation artifact; use target-only "
                "activations with shape [N, D]."
            )
        raise ValueError(
            f"{artifact_label}[{state_name!r}] must have shape [N, D], got {tuple(tensor.shape)}."
        )
    if int(tensor.shape[0]) != num_samples:
        raise ValueError(
            f"{artifact_label}[{state_name!r}] dim-0 {int(tensor.shape[0])} "
            f"!= num_samples {num_samples}."
        )
    if int(tensor.shape[1]) <= 0:
        raise ValueError(f"{artifact_label}[{state_name!r}] has invalid feature dim {tensor.shape[1]}.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{artifact_label}[{state_name!r}] contains NaN or Inf values.")
    return tensor.to(dtype=torch.float64, copy=False)


def prepare_selected_states(
    artifact: ActivationArtifact,
    state_names: Sequence[str],
    *,
    artifact_label: str,
) -> Dict[str, torch.Tensor]:
    """Validate and materialize the selected states as float64 matrices."""

    available = set(artifact.states.keys())
    missing = [state_name for state_name in state_names if state_name not in available]
    if missing:
        raise ValueError(
            f"{artifact_label} is missing required states: {missing}. "
            f"Available states: {sort_state_names(artifact.states.keys())}"
        )

    prepared: Dict[str, torch.Tensor] = {}
    for state_name in state_names:
        prepared[state_name] = _validate_selected_state_tensor(
            artifact.states[state_name],
            state_name=state_name,
            artifact_label=artifact_label,
            num_samples=artifact.num_samples,
        )
    return prepared


def _parse_states_arg(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    state_names = [chunk.strip() for chunk in raw.split(",")]
    if not state_names or any(not name for name in state_names):
        raise ValueError("--states must be a comma-separated list of non-empty state names.")
    if len(set(state_names)) != len(state_names):
        raise ValueError("--states must not contain duplicates.")
    return state_names


def _resolve_mode(mode: Optional[str], artifact_b_supplied: bool) -> str:
    if mode is not None:
        return mode
    return "cross_same_layer" if artifact_b_supplied else "within"


def resolve_state_lists(
    *,
    mode: str,
    artifact_a: ActivationArtifact,
    artifact_b: Optional[ActivationArtifact],
    requested_states: Optional[Sequence[str]],
) -> tuple[List[str], List[str]]:
    """Choose the ordered state lists for the requested comparison mode."""

    available_a = sort_state_names(artifact_a.states.keys())
    available_b = sort_state_names(artifact_b.states.keys()) if artifact_b is not None else []

    if mode == "within":
        states = list(requested_states) if requested_states is not None else available_a
        return states, states

    if artifact_b is None:
        raise ValueError(f"mode={mode!r} requires --artifact-b.")

    if mode == "cross_same_layer":
        states = list(requested_states) if requested_states is not None else available_a
        missing_a = [state_name for state_name in states if state_name not in set(available_a)]
        missing_b = [state_name for state_name in states if state_name not in set(available_b)]
        if missing_a:
            raise ValueError(f"artifact_a is missing states required by --mode cross_same_layer: {missing_a}")
        if missing_b:
            raise ValueError(f"artifact_b is missing states required by --mode cross_same_layer: {missing_b}")
        return states, states

    if mode == "cross_all":
        if requested_states is not None:
            states = list(requested_states)
            missing_a = [state_name for state_name in states if state_name not in set(available_a)]
            missing_b = [state_name for state_name in states if state_name not in set(available_b)]
            if missing_a:
                raise ValueError(f"artifact_a is missing states required by --states: {missing_a}")
            if missing_b:
                raise ValueError(f"artifact_b is missing states required by --states: {missing_b}")
            return states, states
        return available_a, available_b

    raise ValueError(f"Unsupported mode={mode!r}.")


def center_gram_matrix(gram: torch.Tensor) -> torch.Tensor:
    """Center a Gram matrix across the sample axis.

    Parameters
    ----------
    gram:
        Tensor of shape ``[N, N]`` where ``gram[i, j] = <x_i, x_j>``.
    """

    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(f"Gram matrix must have shape [N, N], got {tuple(gram.shape)}.")
    row_mean = gram.mean(dim=1, keepdim=True)
    col_mean = gram.mean(dim=0, keepdim=True)
    grand_mean = gram.mean()
    return gram - row_mean - col_mean + grand_mean


def unbiased_hsic(gram_x: torch.Tensor, gram_y: torch.Tensor) -> torch.Tensor:
    """Unbiased HSIC estimator for two Gram matrices.

    This follows the standard unbiased estimator for HSIC using Gram matrices
    with zeroed diagonals. The result can be slightly negative for finite
    samples, which is expected for an unbiased estimator.
    """

    if gram_x.shape != gram_y.shape:
        raise ValueError(
            f"Unbiased HSIC requires equal Gram shapes, got {tuple(gram_x.shape)} and {tuple(gram_y.shape)}."
        )
    n = int(gram_x.shape[0])
    if n < 4:
        raise ValueError("--unbiased requires at least 4 samples.")

    gram_x = gram_x.clone()
    gram_y = gram_y.clone()
    gram_x.fill_diagonal_(0.0)
    gram_y.fill_diagonal_(0.0)

    trace_term = torch.sum(gram_x * gram_y)
    sum_x = gram_x.sum()
    sum_y = gram_y.sum()
    row_sums_x = gram_x.sum(dim=0)
    row_sums_y = gram_y.sum(dim=0)
    cross_term = torch.dot(row_sums_x, row_sums_y)

    normalizer = float(n * (n - 3))
    correction = (sum_x * sum_y) / float((n - 1) * (n - 2))
    bias_term = (2.0 * cross_term) / float(n - 2)
    return (trace_term + correction - bias_term) / normalizer


def linear_cka(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float = 1e-8,
    centering: str = "gram",
    unbiased: bool = False,
) -> float:
    """Compute linear CKA between two sample-aligned representation matrices.

    Parameters
    ----------
    x, y:
        Two tensors with shapes ``[N, D_x]`` and ``[N, D_y]``. Rows must be the
        same samples in the same order.
    eps:
        Numerical stabilizer used when normalizing.
    centering:
        ``"gram"`` centers linear-kernel Gram matrices. ``"feature"`` centers
        features across the sample dimension before using the equivalent
        feature-space linear CKA formula.
    unbiased:
        When ``True``, use the unbiased HSIC estimator. This is currently
        supported only for ``centering="gram"``.
    """

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"linear_cka expects 2D inputs, got {tuple(x.shape)} and {tuple(y.shape)}.")
    if int(x.shape[0]) != int(y.shape[0]):
        raise ValueError(
            f"linear_cka requires the same number of samples, got {int(x.shape[0])} and {int(y.shape[0])}."
        )
    n = int(x.shape[0])
    if n < 2:
        raise ValueError("linear_cka requires at least 2 samples.")
    if unbiased and centering != "gram":
        raise ValueError("--unbiased is only supported with --centering gram.")

    x = x.to(dtype=torch.float64, copy=False)
    y = y.to(dtype=torch.float64, copy=False)

    if centering == "feature":
        x_centered = x - x.mean(dim=0, keepdim=True)
        y_centered = y - y.mean(dim=0, keepdim=True)
        cross_cov = x_centered.T @ y_centered
        x_cov = x_centered.T @ x_centered
        y_cov = y_centered.T @ y_centered
        numerator = torch.sum(cross_cov.square())
        denom = torch.sqrt(torch.sum(x_cov.square()) * torch.sum(y_cov.square()))
        if float(denom.item()) <= eps:
            raise ValueError("Degenerate representation encountered: feature-space linear CKA denominator is ~0.")
        return float((numerator / denom).item())

    if centering != "gram":
        raise ValueError(f"Unsupported centering={centering!r}. Use 'gram' or 'feature'.")

    gram_x = x @ x.T
    gram_y = y @ y.T
    if unbiased:
        hsic_xy = unbiased_hsic(gram_x, gram_y)
        hsic_xx = unbiased_hsic(gram_x, gram_x)
        hsic_yy = unbiased_hsic(gram_y, gram_y)
        if float(hsic_xx.item()) <= eps or float(hsic_yy.item()) <= eps:
            raise ValueError("Degenerate representation encountered: unbiased linear CKA self-HSIC is ~0.")
        denom = torch.sqrt(hsic_xx * hsic_yy)
        return float((hsic_xy / denom).item())

    gram_x = center_gram_matrix(gram_x)
    gram_y = center_gram_matrix(gram_y)
    numerator = torch.sum(gram_x * gram_y)
    denom = torch.sqrt(torch.sum(gram_x.square()) * torch.sum(gram_y.square()))
    if float(denom.item()) <= eps:
        raise ValueError("Degenerate representation encountered: Gram-space linear CKA denominator is ~0.")
    return float((numerator / denom).item())


def compute_within_cka(
    state_dict: Mapping[str, torch.Tensor],
    state_names: Sequence[str],
    *,
    eps: float,
    centering: str,
    unbiased: bool,
) -> torch.Tensor:
    """Compute a symmetric within-artifact CKA matrix of shape ``[K, K]``."""

    matrix = torch.empty((len(state_names), len(state_names)), dtype=torch.float64)
    for row_idx, row_state in enumerate(state_names):
        matrix[row_idx, row_idx] = linear_cka(
            state_dict[row_state],
            state_dict[row_state],
            eps=eps,
            centering=centering,
            unbiased=unbiased,
        )
        for col_idx in range(row_idx + 1, len(state_names)):
            col_state = state_names[col_idx]
            score = linear_cka(
                state_dict[row_state],
                state_dict[col_state],
                eps=eps,
                centering=centering,
                unbiased=unbiased,
            )
            matrix[row_idx, col_idx] = score
            matrix[col_idx, row_idx] = score
    return matrix


def compute_cross_same_layer_cka(
    state_dict_a: Mapping[str, torch.Tensor],
    state_dict_b: Mapping[str, torch.Tensor],
    state_names: Sequence[str],
    *,
    eps: float,
    centering: str,
    unbiased: bool,
) -> tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    """Compute same-layer cross-artifact CKA scores."""

    vector = torch.empty((len(state_names),), dtype=torch.float64)
    matrix = torch.full((len(state_names), len(state_names)), float("nan"), dtype=torch.float64)
    scores: Dict[str, float] = {}
    for idx, state_name in enumerate(state_names):
        score = linear_cka(
            state_dict_a[state_name],
            state_dict_b[state_name],
            eps=eps,
            centering=centering,
            unbiased=unbiased,
        )
        vector[idx] = score
        matrix[idx, idx] = score
        scores[state_name] = score
    return vector, scores, matrix


def compute_cross_all_cka(
    state_dict_a: Mapping[str, torch.Tensor],
    state_dict_b: Mapping[str, torch.Tensor],
    row_states: Sequence[str],
    col_states: Sequence[str],
    *,
    eps: float,
    centering: str,
    unbiased: bool,
) -> torch.Tensor:
    """Compute the full cross-artifact CKA matrix of shape ``[K_a, K_b]``."""

    matrix = torch.empty((len(row_states), len(col_states)), dtype=torch.float64)
    for row_idx, row_state in enumerate(row_states):
        for col_idx, col_state in enumerate(col_states):
            matrix[row_idx, col_idx] = linear_cka(
                state_dict_a[row_state],
                state_dict_b[col_state],
                eps=eps,
                centering=centering,
                unbiased=unbiased,
            )
    return matrix


def _validate_within_matrix(matrix: torch.Tensor) -> None:
    if matrix.numel() == 0:
        raise ValueError("Within-mode CKA matrix is empty.")
    diag = torch.diag(matrix)
    if not torch.allclose(diag, torch.ones_like(diag), atol=1e-4, rtol=1e-4):
        raise ValueError(
            "Within-mode CKA sanity check failed: diagonal is not close to 1. "
            f"Observed diagonal={diag.tolist()}"
        )
    if not torch.allclose(matrix, matrix.T, atol=1e-6, rtol=1e-6):
        raise ValueError("Within-mode CKA matrix sanity check failed: matrix is not symmetric.")


def _result_structure_for_mode(mode: str) -> str:
    """Return a short description of the saved result layout for a mode."""

    if mode == "within":
        return "full_square_within_model_matrix"
    if mode == "cross_same_layer":
        return "same_layer_scores_vector_dict_plus_diag_only_matrix"
    if mode == "cross_all":
        return "full_cross_artifact_matrix"
    raise ValueError(f"Unsupported mode={mode!r}.")


def _matrix_semantics_for_mode(mode: str) -> str:
    """Return a short description of how to interpret the saved matrix field."""

    if mode == "within":
        return "full_square_within_model_matrix"
    if mode == "cross_same_layer":
        return "diag_only_same_layer_matrix"
    if mode == "cross_all":
        return "full_cross_artifact_matrix"
    raise ValueError(f"Unsupported mode={mode!r}.")


def _same_layer_semantics_for_mode(mode: str) -> Optional[str]:
    """Return an optional description for same-layer outputs."""

    if mode == "cross_same_layer":
        return (
            "same_layer_vector and same_layer_scores follow row_states order; "
            "each score compares artifact_a[state] against artifact_b[state] for the same state name"
        )
    return None


def build_result_meta(
    *,
    mode: str,
    artifact_a: ActivationArtifact,
    artifact_b: Optional[ActivationArtifact],
    row_states: Sequence[str],
    col_states: Sequence[str],
    group_filter: Optional[str],
    sample_limit: Optional[int],
    num_samples_before_filter: int,
    centering: str,
    unbiased: bool,
    eps: float,
    reordered_artifact_b: bool,
) -> Dict[str, Any]:
    return {
        "mode": mode,
        "artifact_a": str(artifact_a.path),
        "artifact_b": None if artifact_b is None else str(artifact_b.path),
        "artifact_a_state_collection": artifact_a.state_collection_name,
        "artifact_b_state_collection": (
            None if artifact_b is None else artifact_b.state_collection_name
        ),
        "group_filter": group_filter,
        "sample_limit": sample_limit,
        "num_samples_before_filter": num_samples_before_filter,
        "num_samples_used": artifact_a.num_samples,
        "states_a": list(row_states),
        "states_b": list(col_states),
        "result_structure": _result_structure_for_mode(mode),
        "matrix_semantics": _matrix_semantics_for_mode(mode),
        "same_layer_semantics": _same_layer_semantics_for_mode(mode),
        "cka_variant": "linear_cka",
        "centering": centering,
        "unbiased": unbiased,
        "eps": eps,
        "reordered_artifact_b": reordered_artifact_b,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script_version": _SCRIPT_VERSION,
    }


def save_results(results: Mapping[str, Any], output_path: Path, overwrite: bool) -> None:
    """Save the CKA result payload as ``.pt``."""

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(results), output_path)


def _run_self_test() -> None:
    """Run lightweight internal sanity checks without reading artifact files."""

    logger.info("running self-test")

    x = torch.tensor(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 3.0],
            [2.0, 1.0, 0.0],
            [3.0, 2.0, 1.0],
        ],
        dtype=torch.float64,
    )
    score_xx = linear_cka(x, x, centering="gram")
    if abs(score_xx - 1.0) > 1e-8:
        raise AssertionError(f"Self-test failed: CKA(X, X)={score_xx} is not close to 1.")

    state_dict = {
        "resid_0": x,
        "resid_1": 2.0 * x,
        "resid_final": x + 1.0,
    }
    within_matrix = compute_within_cka(
        state_dict,
        ["resid_0", "resid_1", "resid_final"],
        eps=1e-8,
        centering="gram",
        unbiased=False,
    )
    _validate_within_matrix(within_matrix)

    sample_ids = ["s0", "s1", "s2", "s3"]
    target_positions = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    artifact_a = ActivationArtifact(
        path=Path("/tmp/compute_cka_self_test_a.pt"),
        meta={},
        sample_ids=sample_ids,
        group_labels=["g0", "g1", "g0", "g1"],
        target_positions=target_positions,
        states={"resid_0": x.clone()},
        state_collection_name=_TARGET_STATE_COLLECTION,
    )
    reorder = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    artifact_b = ActivationArtifact(
        path=Path("/tmp/compute_cka_self_test_b.pt"),
        meta={},
        sample_ids=[sample_ids[idx] for idx in reorder.tolist()],
        group_labels=[artifact_a.group_labels[idx] for idx in reorder.tolist()],
        target_positions=target_positions.index_select(0, reorder),
        states={"resid_0": x.index_select(0, reorder)},
        state_collection_name=_TARGET_STATE_COLLECTION,
    )
    _, aligned_b, reordered = align_artifacts_by_sample_ids(artifact_a, artifact_b)
    if not reordered:
        raise AssertionError("Self-test failed: artifact_b reorder was expected but did not occur.")
    if aligned_b.sample_ids != artifact_a.sample_ids:
        raise AssertionError("Self-test failed: sample_ids were not aligned correctly after reorder.")

    logger.info("self-test passed")
    print("[compute_cka] self-test passed", flush=True)
    print(f"[compute_cka] cka_xx: {score_xx:.6f}", flush=True)
    print(f"[compute_cka] within_matrix_shape: {list(within_matrix.shape)}", flush=True)
    print("[compute_cka] sample_id_reorder: ok", flush=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute linear CKA from activation artifacts exported by extract_residuals.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--artifact-a",
        type=str,
        default=None,
        help="Path to the first activation artifact.",
    )
    parser.add_argument(
        "--artifact-b",
        type=str,
        default=None,
        help="Optional second activation artifact. Required for cross-artifact modes.",
    )
    parser.add_argument("--output", type=str, default=None, help="Output result path (.pt).")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["within", "cross_same_layer", "cross_all"],
        help="Comparison mode. Defaults to 'within' without --artifact-b, otherwise 'cross_same_layer'.",
    )
    parser.add_argument(
        "--states",
        type=str,
        default=None,
        help="Optional comma-separated state list. If omitted, states are inferred and sorted automatically.",
    )
    parser.add_argument(
        "--group-filter",
        type=str,
        default=None,
        help="Optional group label filter applied after sample alignment.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional limit on the number of aligned samples kept after filtering.",
    )
    parser.add_argument(
        "--centering",
        type=str,
        default="gram",
        choices=["gram", "feature"],
        help="Centering strategy for linear CKA (default: gram).",
    )
    parser.add_argument(
        "--unbiased",
        action="store_true",
        help="Use the unbiased HSIC estimator. Currently supported only with --centering gram and N >= 4.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Numerical stability term used during normalization (default: 1e-8).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a lightweight internal sanity check and exit.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite --output if it already exists.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")

    args = parser.parse_args(argv)
    if args.sample_limit is not None and args.sample_limit <= 0:
        parser.error("--sample-limit must be a positive integer.")
    if args.eps <= 0:
        parser.error("--eps must be positive.")
    if not args.self_test and args.artifact_a is None:
        parser.error("--artifact-a is required unless --self-test is used.")
    if not args.self_test and args.output is None:
        parser.error("--output is required unless --self-test is used.")
    return args


def _log_state_list(header: str, state_names: Sequence[str]) -> None:
    logger.info("%s", header)
    for state_name in state_names:
        logger.info("  - %s", state_name)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(name)s] %(message)s",
    )

    if args.self_test:
        _run_self_test()
        return

    requested_states = _parse_states_arg(args.states)
    mode = _resolve_mode(args.mode, artifact_b_supplied=(args.artifact_b is not None))

    if mode == "within" and args.artifact_b is not None:
        raise ValueError("--mode within does not accept --artifact-b.")
    if mode != "within" and args.artifact_b is None:
        raise ValueError(f"--mode {mode} requires --artifact-b.")

    artifact_a = load_artifact(args.artifact_a)
    artifact_b = None if args.artifact_b is None else load_artifact(args.artifact_b)

    logger.info("mode: %s", mode)
    logger.info("artifact_a: %s", artifact_a.path)
    logger.info("artifact_b: %s", None if artifact_b is None else artifact_b.path)
    logger.info("artifact_a state collection: %s", artifact_a.state_collection_name)
    if artifact_b is not None:
        logger.info("artifact_b state collection: %s", artifact_b.state_collection_name)

    reordered_artifact_b = False
    if artifact_b is not None:
        artifact_a, artifact_b, reordered_artifact_b = align_artifacts_by_sample_ids(artifact_a, artifact_b)

    num_samples_before_filter = artifact_a.num_samples
    logger.info("num_samples_before_filter: %d", num_samples_before_filter)
    logger.info("group_filter: %s", args.group_filter)

    artifact_a, artifact_b = apply_group_filter(artifact_a, artifact_b, args.group_filter)
    artifact_a, artifact_b = apply_sample_limit(artifact_a, artifact_b, args.sample_limit)

    if artifact_a.num_samples < 2:
        raise ValueError(
            f"Need at least 2 samples after filtering, got {artifact_a.num_samples}."
        )
    if args.unbiased and artifact_a.num_samples < 4:
        raise ValueError("--unbiased requires at least 4 samples after filtering.")

    logger.info("num_samples_used: %d", artifact_a.num_samples)

    row_states, col_states = resolve_state_lists(
        mode=mode,
        artifact_a=artifact_a,
        artifact_b=artifact_b,
        requested_states=requested_states,
    )

    if mode == "cross_all" and row_states != col_states:
        _log_state_list("states_a:", row_states)
        _log_state_list("states_b:", col_states)
        logger.info("cross_all state counts: rows=%d cols=%d", len(row_states), len(col_states))
    else:
        _log_state_list("states:", row_states)
        if mode == "cross_same_layer":
            logger.info("same-layer state count: %d", len(row_states))

    prepared_a = prepare_selected_states(artifact_a, row_states, artifact_label="artifact_a")
    prepared_b = None
    if artifact_b is not None:
        states_for_b = row_states if mode == "cross_same_layer" else col_states
        prepared_b = prepare_selected_states(artifact_b, states_for_b, artifact_label="artifact_b")
        if artifact_a.num_samples != artifact_b.num_samples:
            raise ValueError(
                f"Aligned artifacts disagree on sample count: {artifact_a.num_samples} vs {artifact_b.num_samples}."
            )

    output_path = Path(args.output).expanduser().resolve()
    result_meta = build_result_meta(
        mode=mode,
        artifact_a=artifact_a,
        artifact_b=artifact_b,
        row_states=row_states,
        col_states=col_states,
        group_filter=args.group_filter,
        sample_limit=args.sample_limit,
        num_samples_before_filter=num_samples_before_filter,
        centering=args.centering,
        unbiased=args.unbiased,
        eps=args.eps,
        reordered_artifact_b=reordered_artifact_b,
    )

    results: Dict[str, Any] = {
        "meta": result_meta,
        "row_states": list(row_states),
        "col_states": list(col_states),
        "sample_ids_used": list(artifact_a.sample_ids),
        "group_labels_used": list(artifact_a.group_labels),
        "target_positions_used": artifact_a.target_positions.clone(),
    }

    if mode == "within":
        matrix = compute_within_cka(
            prepared_a,
            row_states,
            eps=args.eps,
            centering=args.centering,
            unbiased=args.unbiased,
        )
        _validate_within_matrix(matrix)
        results["matrix"] = matrix.cpu()
        save_results(results, output_path=output_path, overwrite=args.overwrite)
        logger.info("matrix shape: %s", list(matrix.shape))

    elif mode == "cross_same_layer":
        assert prepared_b is not None
        same_layer_vector, same_layer_scores, matrix = compute_cross_same_layer_cka(
            prepared_a,
            prepared_b,
            row_states,
            eps=args.eps,
            centering=args.centering,
            unbiased=args.unbiased,
        )
        results["matrix"] = matrix.cpu()
        results["same_layer_vector"] = same_layer_vector.cpu()
        results["same_layer_scores"] = same_layer_scores
        save_results(results, output_path=output_path, overwrite=args.overwrite)
        logger.info("same_layer_vector shape: %s", list(same_layer_vector.shape))
        logger.info("same_layer state count: %d", len(row_states))

    elif mode == "cross_all":
        assert prepared_b is not None
        matrix = compute_cross_all_cka(
            prepared_a,
            prepared_b,
            row_states,
            col_states,
            eps=args.eps,
            centering=args.centering,
            unbiased=args.unbiased,
        )
        if matrix.numel() == 0:
            raise ValueError("Cross-all CKA matrix is empty.")
        results["matrix"] = matrix.cpu()
        save_results(results, output_path=output_path, overwrite=args.overwrite)
        logger.info("matrix shape: %s", list(matrix.shape))
        logger.info("cross_all state counts: rows=%d cols=%d", len(row_states), len(col_states))

    else:
        raise AssertionError(f"Unhandled mode: {mode}")

    logger.info("saved to: %s", output_path)
    print(flush=True)
    print(f"[compute_cka] mode: {mode}", flush=True)
    print(f"[compute_cka] artifact_a: {artifact_a.path}", flush=True)
    print(f"[compute_cka] artifact_b: {None if artifact_b is None else artifact_b.path}", flush=True)
    print(f"[compute_cka] num_samples_before_filter: {num_samples_before_filter}", flush=True)
    print(f"[compute_cka] group_filter: {args.group_filter}", flush=True)
    print(f"[compute_cka] num_samples_used: {artifact_a.num_samples}", flush=True)
    if mode == "cross_all" and row_states != col_states:
        print("[compute_cka] states_a:", flush=True)
        for state_name in row_states:
            print(f"  - {state_name}", flush=True)
        print("[compute_cka] states_b:", flush=True)
        for state_name in col_states:
            print(f"  - {state_name}", flush=True)
    else:
        print("[compute_cka] states:", flush=True)
        for state_name in row_states:
            print(f"  - {state_name}", flush=True)
        if mode == "cross_same_layer":
            print(f"[compute_cka] same_layer_state_count: {len(row_states)}", flush=True)
    print(f"[compute_cka] saved to: {output_path}", flush=True)
    if "matrix" in results:
        print(f"[compute_cka] matrix shape: {list(results['matrix'].shape)}", flush=True)
    if mode == "cross_all":
        print(f"[compute_cka] cross_all_row_state_count: {len(row_states)}", flush=True)
        print(f"[compute_cka] cross_all_col_state_count: {len(col_states)}", flush=True)
    if "same_layer_vector" in results:
        print(f"[compute_cka] same_layer_vector shape: {list(results['same_layer_vector'].shape)}", flush=True)


if __name__ == "__main__":
    main()
