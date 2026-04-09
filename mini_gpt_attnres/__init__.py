"""Mini GPT project with a standard baseline and an AttnRes variant."""

from .config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig, default_experiment
from .model import AttnResGPT, StandardGPT, build_model

__all__ = [
    "AttnResGPT",
    "StandardGPT",
    "DataConfig",
    "ExperimentConfig",
    "ModelConfig",
    "TrainConfig",
    "build_model",
    "default_experiment",
]
