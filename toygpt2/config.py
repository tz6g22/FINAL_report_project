"""Configuration helpers for the toygpt2 project."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class ModelConfig:
    """Model hyperparameters shared by both GPT variants."""

    model_type: str = "standard"
    vocab_size: int = 64
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    bias: bool = True
    attnres_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        if self.model_type not in {"standard", "attnres"}:
            raise ValueError("model_type must be 'standard' or 'attnres'.")


@dataclass
class DataConfig:
    """Dataset settings for synthetic tasks and TinyStories training."""

    dataset_type: str = "tinystories"
    train_size: int = 1024
    val_size: int = 256
    pattern_length: int = 8
    retrieval_pairs: int = 4
    hf_dataset_name: str = "roneneldan/TinyStories"
    text_field: str = "text"
    tokenizer_name: str = "gpt2"
    train_texts: int | None = None
    val_texts: int | None = None
    block_stride: int = 256
    use_token_cache: bool = True
    token_cache_dir: str = "toygpt2_cache/tinystories"

    def __post_init__(self) -> None:
        valid = {"random", "repeated_pattern", "retrieval", "tinystories"}
        if self.dataset_type not in valid:
            raise ValueError(f"dataset_type must be one of {sorted(valid)}.")
        if self.train_texts is not None and self.train_texts <= 0:
            raise ValueError("train_texts must be positive when provided.")
        if self.val_texts is not None and self.val_texts <= 0:
            raise ValueError("val_texts must be positive when provided.")
        if self.block_stride <= 0:
            raise ValueError("block_stride must be positive.")
        if self.use_token_cache and not str(self.token_cache_dir).strip():
            raise ValueError("token_cache_dir must be non-empty when use_token_cache is enabled.")


@dataclass
class TrainConfig:
    """Optimization and checkpointing settings."""

    batch_size: int = 32
    max_steps: int = 20000
    eval_interval: int = 500
    checkpoint_interval: int = 1000
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    eval_batches: int | None = None
    seed: int = 1234
    device: str = "auto"
    num_workers: int = 0
    out_dir: str = "runs/default"
    show_progress: bool = True
    log_every_step: bool = True


@dataclass
class ExperimentConfig:
    """Full experiment config for training or evaluation."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExperimentConfig":
        return cls(
            model=ModelConfig(**payload.get("model", {})),
            data=DataConfig(**payload.get("data", {})),
            train=TrainConfig(**payload.get("train", {})),
        )

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)


def default_experiment(model_type: str = "standard") -> ExperimentConfig:
    """Return the default TinyStories training configuration."""

    experiment = ExperimentConfig()
    experiment.model.model_type = model_type
    experiment.train.out_dir = f"toygpt2_runs/{model_type}"
    return experiment
