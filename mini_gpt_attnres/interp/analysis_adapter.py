"""Unified trace records for StandardGPT and AttnResGPT analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence

import torch

from .cache import ActivationCache

_BLOCK_KEY_PATTERN = re.compile(r"^blocks\.(\d+)\.")


@dataclass
class AttnResSiteTrace:
    """Trace payload for one AttnRes site."""

    current: torch.Tensor | None = None
    history_states: torch.Tensor | None = None
    scores: torch.Tensor | None = None
    weights: torch.Tensor | None = None
    aggregated: torch.Tensor | None = None
    unavailable: Dict[str, str] = field(default_factory=dict)


@dataclass
class TraceRecord:
    """Layer-level unified record across model variants."""

    model_type: str
    block_type: str
    layer_index: int

    layer_input: torch.Tensor | None = None
    attn_out: torch.Tensor | None = None
    residual_after_attn: torch.Tensor | None = None
    mlp_out: torch.Tensor | None = None
    layer_output: torch.Tensor | None = None

    attn_probs: torch.Tensor | None = None
    q: torch.Tensor | None = None
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None

    attnres_pre_attn: AttnResSiteTrace | None = None
    attnres_pre_mlp: AttnResSiteTrace | None = None

    unavailable: Dict[str, str] = field(default_factory=dict)
    source_keys: Dict[str, str] = field(default_factory=dict)

    @property
    def token_axis(self) -> int:
        """Token position axis in cached tensors."""

        return 1


@dataclass
class ModelTrace:
    """Container returned by AnalysisAdapter."""

    model_type: str
    records: List[TraceRecord]
    cache_keys: Sequence[str]
    metadata: Dict[str, object] = field(default_factory=dict)

    def layer(self, layer_index: int) -> TraceRecord:
        for record in self.records:
            if record.layer_index == layer_index:
                return record
        raise KeyError(f"Layer {layer_index} is not present in trace.")

    def layers(self) -> Iterable[int]:
        return [record.layer_index for record in self.records]


class AnalysisAdapter:
    """Map raw cache/intermediates to a unified analysis schema."""

    @staticmethod
    def from_model_outputs(outputs: Mapping[str, object]) -> ModelTrace:
        """Build a ModelTrace from model(...) outputs."""

        if "intermediates" in outputs:
            intermediates = outputs["intermediates"]
            if not isinstance(intermediates, Mapping):
                raise TypeError("outputs['intermediates'] must be a mapping of cache_key -> tensor.")
            metadata: Dict[str, object] = {}
            cache_wrapper = outputs.get("cache")
            if isinstance(cache_wrapper, ActivationCache):
                metadata.update(cache_wrapper.metadata)
            model_type = str(metadata.get("model_type", outputs.get("model_type", "unknown")))
            return AnalysisAdapter.from_cache_dict(dict(intermediates), model_type=model_type, metadata=metadata)

        if "cache" in outputs and isinstance(outputs["cache"], ActivationCache):
            cache_obj = outputs["cache"]
            return AnalysisAdapter.from_cache_dict(
                dict(cache_obj.data),
                model_type=str(cache_obj.metadata.get("model_type", outputs.get("model_type", "unknown"))),
                metadata=dict(cache_obj.metadata),
            )

        raise KeyError("Model outputs must contain either 'intermediates' or an ActivationCache at 'cache'.")

    @staticmethod
    def from_cache(
        cache: ActivationCache | Mapping[str, torch.Tensor],
        model_type: str = "unknown",
        metadata: Mapping[str, object] | None = None,
    ) -> ModelTrace:
        if isinstance(cache, ActivationCache):
            merged_metadata = dict(cache.metadata)
            if metadata is not None:
                merged_metadata.update(dict(metadata))
            mt = str(merged_metadata.get("model_type", model_type))
            return AnalysisAdapter.from_cache_dict(dict(cache.data), model_type=mt, metadata=merged_metadata)
        return AnalysisAdapter.from_cache_dict(dict(cache), model_type=model_type, metadata=metadata)

    @staticmethod
    def from_cache_dict(
        cache: Mapping[str, torch.Tensor],
        model_type: str = "unknown",
        metadata: Mapping[str, object] | None = None,
    ) -> ModelTrace:
        """Build a ModelTrace directly from a cache dict."""

        layer_indices = AnalysisAdapter._extract_layer_indices(cache.keys())
        records = [AnalysisAdapter._build_record(cache, model_type, layer_idx) for layer_idx in layer_indices]
        return ModelTrace(
            model_type=model_type,
            records=records,
            cache_keys=sorted(cache.keys()),
            metadata={} if metadata is None else dict(metadata),
        )

    @staticmethod
    def _extract_layer_indices(keys: Iterable[str]) -> List[int]:
        layer_indices = set()
        for key in keys:
            matched = _BLOCK_KEY_PATTERN.match(key)
            if matched:
                layer_indices.add(int(matched.group(1)))
        return sorted(layer_indices)

    @staticmethod
    def _build_record(cache: Mapping[str, torch.Tensor], model_type: str, layer_idx: int) -> TraceRecord:
        prefix = f"blocks.{layer_idx}"
        has_attnres_keys = any(key.startswith(f"{prefix}.attnres.") for key in cache)
        block_type = "attnres" if has_attnres_keys or model_type == "attnres" else "standard"

        record = TraceRecord(model_type=model_type, block_type=block_type, layer_index=layer_idx)

        def fetch(field_name: str, key: str) -> torch.Tensor | None:
            if key in cache:
                record.source_keys[field_name] = key
                return cache[key]
            record.unavailable[field_name] = f"missing cache key: {key}"
            return None

        record.layer_input = fetch("layer_input", f"{prefix}.input")
        record.attn_out = fetch("attn_out", f"{prefix}.attn_out")
        record.mlp_out = fetch("mlp_out", f"{prefix}.mlp_out")
        record.layer_output = fetch("layer_output", f"{prefix}.output")
        record.attn_probs = fetch("attn_probs", f"{prefix}.attn_probs")
        record.q = fetch("q", f"{prefix}.q")
        record.k = fetch("k", f"{prefix}.k")
        record.v = fetch("v", f"{prefix}.v")

        if block_type == "standard":
            if record.layer_input is not None and record.attn_out is not None:
                record.residual_after_attn = record.layer_input + record.attn_out
            else:
                record.unavailable["residual_after_attn"] = (
                    "requires both layer_input and attn_out to reconstruct standard residual."
                )
            record.unavailable["attnres_pre_attn"] = "not_applicable_for_standard_block"
            record.unavailable["attnres_pre_mlp"] = "not_applicable_for_standard_block"
            return record

        record.unavailable["residual_after_attn"] = (
            "attnres uses site aggregation; no direct standard residual_after_attn field."
        )
        record.attnres_pre_attn = AnalysisAdapter._build_site_trace(cache, f"{prefix}.attnres.pre_attn")
        record.attnres_pre_mlp = AnalysisAdapter._build_site_trace(cache, f"{prefix}.attnres.pre_mlp")
        return record

    @staticmethod
    def _build_site_trace(cache: Mapping[str, torch.Tensor], site_prefix: str) -> AttnResSiteTrace:
        site = AttnResSiteTrace()

        def fetch(suffix: str) -> torch.Tensor | None:
            key = f"{site_prefix}.{suffix}"
            if key in cache:
                return cache[key]
            site.unavailable[suffix] = f"missing cache key: {key}"
            return None

        site.current = fetch("current")
        site.history_states = fetch("history_states")
        site.scores = fetch("scores")
        site.weights = fetch("weights")
        site.aggregated = fetch("aggregated")
        return site

