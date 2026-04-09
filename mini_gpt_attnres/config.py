"""Configuration helpers for the mini GPT AttnRes project."""

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
    block_size: int = 64
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
    """Synthetic language-modeling dataset settings."""

    dataset_type: str = "repeated_pattern"
    train_size: int = 1024
    val_size: int = 256
    pattern_length: int = 8
    retrieval_pairs: int = 4
    hf_dataset_name: str = "roneneldan/TinyStories"
    text_field: str = "text"
    tokenizer_name: str = "gpt2"
    train_texts: int = 2000
    val_texts: int = 200
    block_stride: int = 64

    def __post_init__(self) -> None:
        valid = {"random", "repeated_pattern", "retrieval", "tinystories"}
        if self.dataset_type not in valid:
            raise ValueError(f"dataset_type must be one of {sorted(valid)}.")


@dataclass
class TrainConfig:
    """Optimization and checkpointing settings."""

    batch_size: int = 32
    max_steps: int = 200
    eval_interval: int = 25
    checkpoint_interval: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    eval_batches: int = 10
    seed: int = 1234
    device: str = "auto"
    num_workers: int = 0
    out_dir: str = "runs/default"


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
    """Return a small default config for smoke tests and toy runs."""

    experiment = ExperimentConfig()
    experiment.model.model_type = model_type
    experiment.train.out_dir = f"runs/{model_type}"
    return experiment
