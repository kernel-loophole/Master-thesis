"""Sparse Autoencoder (SAE) model, loss, trainer, metrics, and feature analysis."""

from spatial_sae.sae.model import SparseAutoencoder
from spatial_sae.sae.loss import SAELoss
from spatial_sae.sae.metrics import compute_sae_metrics
from spatial_sae.sae.trainer import SAETrainer
from spatial_sae.sae.feature_analysis import (
    get_top_activating_examples,
    feature_label_correlations,
    feature_attribute_correlations,
    rank_features_for_concept,
)

__all__ = [
    "SparseAutoencoder",
    "SAELoss",
    "compute_sae_metrics",
    "SAETrainer",
    "get_top_activating_examples",
    "feature_label_correlations",
    "feature_attribute_correlations",
    "rank_features_for_concept",
]
