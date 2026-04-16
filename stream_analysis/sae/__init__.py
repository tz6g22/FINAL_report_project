"""Sparse autoencoder utilities for residual-stream analysis.

This subpackage is intentionally isolated from the existing toy GPT / AttnRes
training and interpretability flows. It reuses the current checkpoint loading,
data-loading, and activation-cache interfaces without modifying their public
behavior.
"""

from .config import SAEConfig, SAEEvalConfig, SAEExtractConfig, SAETrainConfig
from .data import ActivationShardDataset, build_activation_dataloaders, preprocess_activations
from .eval import SAEEvaluator, evaluate_sae_on_loader, load_sae_checkpoint
from .extract import BlockInputExtractor, extract_activation_shards
from .losses import auxk_loss, avg_l0, compute_loss_dict, dead_latent_stats, reconstruction_mse
from .model import TopKActivation, TopKSAE
from .train import SAETrainer
from .analysis import (
    build_comparison_rows,
    compute_feature_mem_auc,
    compute_feature_selectivity,
    load_feature_stats_rows,
    load_mem_labels,
    run_sae_mem_analysis,
    save_comparison_rows,
    select_top_feature_ids,
    summarize_mem_features,
    top_activating_examples,
)
from .intervention import encode_with_error, rebuild_activation, reconstruct_with_feature_edit
from .patching import build_sae_site_override, site_cache_key
from .experiment import (
    compute_target_token_metrics,
    parse_feature_ids,
    run_sae_checkpoint_study,
    run_sae_feature_sweep,
    run_sae_intervention_experiment,
)

__all__ = [
    "ActivationShardDataset",
    "BlockInputExtractor",
    "SAEConfig",
    "SAEEvalConfig",
    "SAEExtractConfig",
    "SAETrainConfig",
    "SAEEvaluator",
    "SAETrainer",
    "TopKActivation",
    "TopKSAE",
    "auxk_loss",
    "avg_l0",
    "build_activation_dataloaders",
    "build_comparison_rows",
    "build_sae_site_override",
    "compute_feature_mem_auc",
    "compute_feature_selectivity",
    "compute_target_token_metrics",
    "compute_loss_dict",
    "dead_latent_stats",
    "encode_with_error",
    "evaluate_sae_on_loader",
    "extract_activation_shards",
    "load_feature_stats_rows",
    "load_mem_labels",
    "load_sae_checkpoint",
    "parse_feature_ids",
    "preprocess_activations",
    "rebuild_activation",
    "reconstruction_mse",
    "reconstruct_with_feature_edit",
    "run_sae_checkpoint_study",
    "run_sae_feature_sweep",
    "run_sae_intervention_experiment",
    "run_sae_mem_analysis",
    "save_comparison_rows",
    "select_top_feature_ids",
    "site_cache_key",
    "summarize_mem_features",
    "top_activating_examples",
]
