#!/usr/bin/env python3
"""CLI Script to run causal steering intervention experiment."""

import argparse
from spatial_sae.utils.config import load_config
from spatial_sae.experiments.intervene import run_causal_interventions
from spatial_sae.experiments.compositional_binding import run_compositional_binding_experiment
from spatial_sae.utils.logging import setup_logging

logger = setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Run Causal Steering Interventions")
    parser.add_argument("--config", type=str, default="configs/intervention/default.yaml", help="Intervention config path")
    parser.add_argument("--compositional", action="store_true", help="Run compositional binding experiment")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.compositional:
        logger.info("Executing Compositional Binding Intervention Experiment...")
        results = run_compositional_binding_experiment(config)
    else:
        logger.info("Executing Causal Steering Intervention Experiment...")
        results = run_causal_interventions(config)

    logger.info("Intervention experiment completed successfully.")


if __name__ == "__main__":
    main()
