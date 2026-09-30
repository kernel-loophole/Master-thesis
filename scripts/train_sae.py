#!/usr/bin/env python3
"""CLI Script to train Sparse Autoencoder (SAE) on extracted activations."""

import argparse
from spatial_sae.utils.config import load_config
from spatial_sae.experiments.train_sae import run_train_sae
from spatial_sae.utils.logging import setup_logging

logger = setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Train Sparse Autoencoder (SAE)")
    parser.add_argument("--config", type=str, default="configs/sae/default.yaml", help="SAE config path")
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info("Executing SAE training CLI job...")
    history = run_train_sae(config)
    logger.info("SAE training job finished successfully.")


if __name__ == "__main__":
    main()
