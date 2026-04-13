"""Unified clean/corrupted patching runner for memorization-style analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Sequence

import torch

from .analysis_adapter import AnalysisAdapter, ModelTrace
from .cache import ActivationCache
from .patching import patch_attention_head_override, patch_from_cache

MetricFn = Callable[[torch.Tensor, int, int], torch.Tensor]


@dataclass
class PatchingRunResult:
    """One clean/corrupted/direct-patch run result."""

    model_type: str
    metric_name: str
    patch_site: str
    target_position: int
    target_token_id: int

    baseline_clean_score: torch.Tensor
    baseline_corrupted_score: torch.Tensor
    patched_score: torch.Tensor
    effect_size: torch.Tensor
    recovery_fraction: torch.Tensor

    position_scores: Dict[str, torch.Tensor]
    decomposition: Dict[str, torch.Tensor]
    metadata: Dict[str, object] = field(default_factory=dict)

    clean_trace: ModelTrace | None = None
    corrupted_trace: ModelTrace | None = None
    patched_trace: ModelTrace | None = None


@dataclass
class SiteSweepResult:
    """Batch result over multiple patch sites."""

    patch_sites: List[str]
    effect_by_site: torch.Tensor
    position_effect_by_site: torch.Tensor
    results: List[PatchingRunResult]
    metadata: Dict[str, object] = field(default_factory=dict)


class MemorizationPatchingRunner:
    """Unified runner for clean/corrupted direct patching experiments."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: object | None = None,
        encoder: Callable[[str], Sequence[int]] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.adapter = AnalysisAdapter()

    def run(
        self,
        clean_input: torch.Tensor | Sequence[int] | str,
        corrupted_input: torch.Tensor | Sequence[int] | str,
        patch_site: str,
        target_position: int,
        target_token_id: int | None = None,
        metric: str | MetricFn = "logit",
        head_index: int | None = None,
    ) -> PatchingRunResult:
        clean_tokens = self._to_tokens(clean_input)
        corrupted_tokens = self._to_tokens(corrupted_input)
        if clean_tokens.shape != corrupted_tokens.shape:
            raise ValueError(
                f"clean and corrupted token shapes must match, got {tuple(clean_tokens.shape)} vs {tuple(corrupted_tokens.shape)}."
            )
        if target_position < 0 or target_position >= clean_tokens.size(1):
            raise IndexError(
                f"target_position {target_position} is out of range for sequence length {clean_tokens.size(1)}."
            )

        resolved_target_token_id = self._resolve_target_token_id(target_token_id, clean_tokens, target_position)
        metric_name, score_fn, position_fn = self._resolve_metric(metric)

        self.model.eval()
        with torch.no_grad():
            clean_outputs = self.model(clean_tokens, return_intermediates=True, return_cache=True)
            corrupted_outputs = self.model(corrupted_tokens, return_intermediates=True, return_cache=True)

        clean_cache = self._extract_cache(clean_outputs)
        corrupted_cache = self._extract_cache(corrupted_outputs)

        if patch_site not in clean_cache:
            raise KeyError(f"patch_site '{patch_site}' is missing from clean cache.")
        if patch_site not in corrupted_cache:
            raise KeyError(f"patch_site '{patch_site}' is missing from corrupted cache.")

        clean_cache_obj = ActivationCache(dict(clean_cache), metadata=self._extract_metadata(clean_outputs))
        if head_index is None:
            overrides = patch_from_cache(clean_cache_obj, patch_site)
        else:
            self._validate_head_index(clean_cache[patch_site], head_index=head_index, patch_site=patch_site)
            overrides = patch_attention_head_override(clean_cache_obj, patch_site=patch_site, head_index=head_index)

        with torch.no_grad():
            patched_outputs = self.model(
                corrupted_tokens,
                return_intermediates=True,
                return_cache=True,
                activation_overrides=overrides,
            )

        clean_logits = clean_outputs["logits"]
        corrupted_logits = corrupted_outputs["logits"]
        patched_logits = patched_outputs["logits"]

        vocab_size = clean_logits.size(-1)
        if resolved_target_token_id >= vocab_size:
            raise IndexError(
                f"target_token_id {resolved_target_token_id} is out of range for vocab_size {vocab_size}."
            )

        clean_score = score_fn(clean_logits, target_position, resolved_target_token_id)
        corrupted_score = score_fn(corrupted_logits, target_position, resolved_target_token_id)
        patched_score = score_fn(patched_logits, target_position, resolved_target_token_id)
        effect_size = patched_score - corrupted_score
        denom = clean_score - corrupted_score
        recovery_fraction = torch.where(
            torch.abs(denom) > 1e-12,
            effect_size / denom,
            torch.full_like(effect_size, float("nan")),
        )

        clean_position_scores = position_fn(clean_logits, resolved_target_token_id)
        corrupted_position_scores = position_fn(corrupted_logits, resolved_target_token_id)
        patched_position_scores = position_fn(patched_logits, resolved_target_token_id)
        position_effect = patched_position_scores - corrupted_position_scores

        model_type = str(self._extract_metadata(clean_outputs).get("model_type", self._infer_model_type()))
        clean_trace = self.adapter.from_model_outputs(clean_outputs)
        corrupted_trace = self.adapter.from_model_outputs(corrupted_outputs)
        patched_trace = self.adapter.from_model_outputs(patched_outputs)

        return PatchingRunResult(
            model_type=model_type,
            metric_name=metric_name,
            patch_site=patch_site,
            target_position=target_position,
            target_token_id=resolved_target_token_id,
            baseline_clean_score=clean_score,
            baseline_corrupted_score=corrupted_score,
            patched_score=patched_score,
            effect_size=effect_size,
            recovery_fraction=recovery_fraction,
            position_scores={
                "clean": clean_position_scores,
                "corrupted": corrupted_position_scores,
                "patched": patched_position_scores,
                "effect": position_effect,
            },
            decomposition={
                "site_effect": effect_size,
                "site_position_effect": position_effect,
            },
            metadata={
                "head_index": head_index,
                "clean_shape": tuple(clean_tokens.shape),
                "patch_tensor_shape": tuple(clean_cache[patch_site].shape),
            },
            clean_trace=clean_trace,
            corrupted_trace=corrupted_trace,
            patched_trace=patched_trace,
        )

    def run_site_sweep(
        self,
        clean_input: torch.Tensor | Sequence[int] | str,
        corrupted_input: torch.Tensor | Sequence[int] | str,
        patch_sites: Sequence[str],
        target_position: int,
        target_token_id: int | None = None,
        metric: str | MetricFn = "logit",
        head_index: int | None = None,
    ) -> SiteSweepResult:
        if not patch_sites:
            raise ValueError("patch_sites must be non-empty.")

        results: List[PatchingRunResult] = []
        for patch_site in patch_sites:
            result = self.run(
                clean_input=clean_input,
                corrupted_input=corrupted_input,
                patch_site=patch_site,
                target_position=target_position,
                target_token_id=target_token_id,
                metric=metric,
                head_index=head_index,
            )
            results.append(result)

        effect_by_site = torch.stack([result.effect_size for result in results], dim=0)
        position_effect_by_site = torch.stack([result.position_scores["effect"] for result in results], dim=0)
        return SiteSweepResult(
            patch_sites=list(patch_sites),
            effect_by_site=effect_by_site,
            position_effect_by_site=position_effect_by_site,
            results=results,
            metadata={
                "metric": results[0].metric_name,
                "target_position": target_position,
                "target_token_id": results[0].target_token_id,
            },
        )

    def _to_tokens(self, value: torch.Tensor | Sequence[int] | str) -> torch.Tensor:
        device = self._model_device()
        if torch.is_tensor(value):
            tokens = value.to(device=device, dtype=torch.long)
            if tokens.ndim == 1:
                return tokens.unsqueeze(0)
            if tokens.ndim == 2:
                return tokens
            raise ValueError(f"Token tensor must have rank 1 or 2, got shape {tuple(tokens.shape)}.")

        if isinstance(value, str):
            token_ids = self._encode_text(value)
            return torch.tensor([token_ids], dtype=torch.long, device=device)

        token_ids = list(value)
        if not token_ids:
            raise ValueError("Input token sequence cannot be empty.")
        return torch.tensor([token_ids], dtype=torch.long, device=device)

    def _encode_text(self, text: str) -> Sequence[int]:
        if self.encoder is not None:
            return list(self.encoder(text))
        if self.tokenizer is None:
            raise ValueError("String inputs require either tokenizer or encoder.")
        if not hasattr(self.tokenizer, "encode"):
            raise ValueError("Provided tokenizer does not expose an encode(...) method.")
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        if not encoded:
            raise ValueError("Tokenizer produced an empty token sequence.")
        return encoded

    def _resolve_target_token_id(
        self,
        target_token_id: int | None,
        clean_tokens: torch.Tensor,
        target_position: int,
    ) -> int:
        if target_token_id is None:
            return int(clean_tokens[0, target_position].item())
        if target_token_id < 0:
            raise ValueError(f"target_token_id must be non-negative, got {target_token_id}.")
        return int(target_token_id)

    def _resolve_metric(self, metric: str | MetricFn) -> tuple[str, MetricFn, Callable[[torch.Tensor, int], torch.Tensor]]:
        if callable(metric):
            return "custom", metric, self._position_from_custom(metric)

        if metric == "logit":
            return (
                "logit",
                lambda logits, pos, token_id: logits[:, pos, token_id],
                lambda logits, token_id: logits[:, :, token_id],
            )
        if metric == "logprob":
            return (
                "logprob",
                lambda logits, pos, token_id: torch.log_softmax(logits, dim=-1)[:, pos, token_id],
                lambda logits, token_id: torch.log_softmax(logits, dim=-1)[:, :, token_id],
            )
        if metric == "prob":
            return (
                "prob",
                lambda logits, pos, token_id: torch.softmax(logits, dim=-1)[:, pos, token_id],
                lambda logits, token_id: torch.softmax(logits, dim=-1)[:, :, token_id],
            )
        raise ValueError(f"Unsupported metric '{metric}'. Valid values are: logit, logprob, prob, or callable.")

    def _position_from_custom(self, metric: MetricFn) -> Callable[[torch.Tensor, int], torch.Tensor]:
        def position_fn(logits: torch.Tensor, token_id: int) -> torch.Tensor:
            values = [metric(logits, pos, token_id) for pos in range(logits.size(1))]
            return torch.stack(values, dim=1)

        return position_fn

    @staticmethod
    def _extract_cache(outputs: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
        if "intermediates" in outputs and isinstance(outputs["intermediates"], Mapping):
            return outputs["intermediates"]
        if "cache" in outputs and isinstance(outputs["cache"], ActivationCache):
            return outputs["cache"].data
        raise KeyError("Outputs do not contain a usable cache/intermediates mapping.")

    @staticmethod
    def _extract_metadata(outputs: Mapping[str, object]) -> Dict[str, object]:
        if "cache" in outputs and isinstance(outputs["cache"], ActivationCache):
            return dict(outputs["cache"].metadata)
        return {}

    @staticmethod
    def _validate_head_index(site_tensor: torch.Tensor, head_index: int, patch_site: str) -> None:
        if site_tensor.ndim < 2:
            raise ValueError(
                f"head_index was provided for patch_site '{patch_site}', but tensor rank is {site_tensor.ndim}."
            )
        max_head = site_tensor.size(1) - 1
        if head_index < 0 or head_index > max_head:
            raise IndexError(f"head_index {head_index} is out of range [0, {max_head}] for patch_site '{patch_site}'.")

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:  # pragma: no cover - defensive path
            return torch.device("cpu")

    def _infer_model_type(self) -> str:
        config = getattr(self.model, "config", None)
        model_type = getattr(config, "model_type", None)
        return "unknown" if model_type is None else str(model_type)
