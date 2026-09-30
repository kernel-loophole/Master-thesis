import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

from spatial_sae.utils.config import ConfigDict, load_config
from spatial_sae.utils.reproducibility import set_seed
from spatial_sae.utils.logging import get_logger, setup_logging
from spatial_sae.sae.model import SparseAutoencoder
from spatial_sae.sae.trainer import SAETrainer
from spatial_sae.data.datasets import ActivationDataset

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    torch = None

logger = get_logger("spatial_sae.exp_train_sae")


def run_train_sae(config: ConfigDict) -> Dict[str, Any]:
    """Execute SAE training experiment according to config."""
    set_seed(config.get("seed", 42))

    output_dir = Path(config.get("output_dir", "./outputs/sae"))
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=output_dir / "train_sae.log")

    input_dim = config.get("input_dim", 512)
    latent_dim = config.get("latent_dim", input_dim * config.get("expansion_factor", 8))

    logger.info(f"Initializing Sparse Autoencoder (Input Dim: {input_dim}, Latent Dim: {latent_dim})")

    sae = SparseAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        activation_function=config.get("activation_function", "relu"),
        top_k=config.get("top_k", 32)
    )

    # Generate synthetic dummy activation dataset if running in dry run / initial setup mode
    if torch is not None:
        dummy_activations = torch.randn(1000, input_dim)
        dataset = ActivationDataset(dummy_activations)
        train_loader = DataLoader(dataset, batch_size=config.get("batch_size", 128), shuffle=True)
    else:
        train_loader = None

    trainer = SAETrainer(
        sae=sae,
        train_loader=train_loader,
        learning_rate=config.get("learning_rate", 3e-4),
        sparsity_coefficient=config.get("sparsity_coefficient", 0.001),
        output_dir=output_dir,
        device=config.get("device", "cpu"),
        use_amp=False
    )

    epochs = config.get("epochs", 5)
    history = trainer.train(epochs=epochs)

    # Save experiment summary and config
    metrics_path = output_dir / "experiment_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Completed SAE training experiment. Output directory: {output_dir}")
    return history
