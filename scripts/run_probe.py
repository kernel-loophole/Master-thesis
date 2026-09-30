#!/usr/bin/env python3
"""CLI Script to train and evaluate Linear Probe baseline."""

import argparse
from spatial_sae.utils.config import load_config
from spatial_sae.experiments.baseline import run_probe_baseline
from spatial_sae.utils.logging import setup_logging

logger = setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Run Linear Probe Baseline")
    parser.add_argument("--config", type=str, default="configs/experiments/default.yaml", help="Experiment config path")
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info("Executing Linear Probe Baseline Experiment...")
    metrics = run_probe_baseline(config)
    logger.info("Linear probe baseline completed successfully.")


if __name__ == "__main__":
    main()
