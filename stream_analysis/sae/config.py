"""Dataclass-based configuration for the SAE subproject."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

_VALID_PREPROCESSING = {
    "none",
    "mean-center",
    "unit-norm",
    "mean-center+unit-norm",
}
_VALID_SITES = {"input", "attn_out", "mlp_out", "output", "final_residual"}


def resolve_preprocessing_mode(
    preprocessing: str | None = "none",
    *,
    input_centering: bool = False,
    input_norm: bool = False,
) -> str:
    """Resolve preprocessing flags into one canonical mode string."""

    base = "none" if preprocessing is None else str(preprocessing).strip().lower()
    if base not in _VALID_PREPROCESSING:
        raise ValueError(
            f"preprocessing must be one of {sorted(_VALID_PREPROCESSING)}, got {preprocessing!r}."
        )

    if input_centering and input_norm:
        derived = "mean-center+unit-norm"
    elif input_centering:
        derived = "mean-center"
    elif input_norm:
        derived = "unit-norm"
    else:
        derived = "none"

    if base != "none" and derived != "none" and base != derived:
        raise ValueError(
            "Conflicting preprocessing settings: "
            f"preprocessing={base!r}, flags_resolve_to={derived!r}."
        )
    return derived if derived != "none" else base


class JsonDataclassMixin:
    """Small helper mixin used by all SAE config dataclasses."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "JsonDataclassMixin":
        return cls(**payload)

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> "JsonDataclassMixin":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)


@dataclass
class SAEConfig(JsonDataclassMixin):
    """Model configuration for a TopK SAE."""

    d_in: int
    n_latents: int
    k: int
    use_auxk: bool = False
    auxk_alpha: float = 0.0
    normalize_decoder: bool = True
    tied_init: bool = False
    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.d_in <= 0:
            raise ValueError(f"d_in must be positive, got {self.d_in}.")
        if self.n_latents <= 0:
            raise ValueError(f"n_latents must be positive, got {self.n_latents}.")
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}.")
        if self.k > self.n_latents:
            raise ValueError(f"k={self.k} cannot exceed n_latents={self.n_latents}.")
        if self.auxk_alpha < 0.0:
            raise ValueError(f"auxk_alpha must be non-negative, got {self.auxk_alpha}.")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be one of: float32, float16, bfloat16.")


@dataclass
class SAETrainConfig(JsonDataclassMixin):
    """Optimization and logging config for SAE training."""

    batch_size: int = 512
    num_steps: int = 10_000
    lr: float = 3e-4
    optimizer: str = "adamw"
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eval_interval: int = 250
    log_interval: int = 25
    checkpoint_interval: int = 500
    val_fraction: float = 0.1
    seed: int = 1234
    num_workers: int = 0
    grad_clip: float | None = 1.0
    dead_threshold: float = 1e-8
    preprocessing: str = "none"
    input_centering: bool = False
    input_norm: bool = False
    max_val_batches: int | None = None
    metrics_filename: str = "metrics.csv"
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive.")
        if self.optimizer not in {"adam", "adamw"}:
            raise ValueError("optimizer must be 'adam' or 'adamw'.")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must lie in [0.0, 1.0).")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive.")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive.")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive.")
        if self.dead_threshold < 0.0:
            raise ValueError("dead_threshold must be non-negative.")
        if self.grad_clip is not None and self.grad_clip <= 0.0:
            raise ValueError("grad_clip must be positive when provided.")
        self.preprocessing = resolve_preprocessing_mode(
            self.preprocessing,
            input_centering=self.input_centering,
            input_norm=self.input_norm,
        )


@dataclass
class SAEExtractConfig(JsonDataclassMixin):
    """Config for activation extraction into SAE shard artifacts."""

    model_type: str
    checkpoint_path: str
    layer_idx: int
    site: str = "input"
    dataset_split: str = "train"
    max_tokens: int | None = None
    out_dir: str = ""
    batch_size: int = 8
    device: str = "auto"
    dtype: str = "float32"
    num_workers: int = 0
    shard_size_tokens: int = 131_072
    save_format: str = "pt"
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.layer_idx < 0:
            raise ValueError(f"layer_idx must be non-negative, got {self.layer_idx}.")
        site = str(self.site).strip().lower()
        if site not in _VALID_SITES:
            raise ValueError(f"site must be one of {sorted(_VALID_SITES)}, got {self.site!r}.")
        self.site = site
        split = str(self.dataset_split).strip().lower()
        if split not in {"train", "val"}:
            raise ValueError("dataset_split must be 'train' or 'val'.")
        self.dataset_split = split
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive when provided.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.shard_size_tokens <= 0:
            raise ValueError("shard_size_tokens must be positive.")
        if self.save_format not in {"pt", "npy"}:
            raise ValueError("save_format must be 'pt' or 'npy'.")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be one of: float32, float16, bfloat16.")


@dataclass
class SAEEvalConfig(JsonDataclassMixin):
    """Config for SAE evaluation on saved activation shards."""

    sae_checkpoint_path: str
    activation_dir: str
    batch_size: int = 512
    device: str = "auto"
    num_workers: int = 0
    preprocessing: str | None = None
    input_centering: bool = False
    input_norm: bool = False
    max_batches: int | None = None
    out_dir: str = ""
    metrics_filename: str = "metrics.json"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.max_batches is not None and self.max_batches <= 0:
            raise ValueError("max_batches must be positive when provided.")
        if self.preprocessing is not None:
            self.preprocessing = resolve_preprocessing_mode(
                self.preprocessing,
                input_centering=self.input_centering,
                input_norm=self.input_norm,
            )
