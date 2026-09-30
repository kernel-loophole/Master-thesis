#!/usr/bin/env python3
"""CLI Script to analyze and discover interpretable SAE spatial features."""

import argparse
from spatial_sae.utils.config import load_config
from spatial_sae.experiments.discover_features import run_discover_features
from spatial_sae.utils.logging import setup_logging

logger = setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Analyze SAE Features and Spatial Concept Correlations")
    parser.add_argument("--config", type=str, default="configs/sae/default.yaml", help="SAE config path")
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info("Executing SAE Feature Discovery and Correlation Analysis...")
    results = run_discover_features(config)
    logger.info("Feature analysis completed successfully.")


if __name__ == "__main__":
    main()
