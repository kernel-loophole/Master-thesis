#!/usr/bin/env python3
"""CLI Script to prepare synthetic spatial reasoning dataset with counterfactual pairs."""

import argparse
from pathlib import Path

from spatial_sae.utils.config import load_config
from spatial_sae.data.generation import SyntheticSpatialDatasetGenerator
from spatial_sae.utils.logging import setup_logging

logger = setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Prepare Spatial Reasoning Synthetic Dataset")
    parser.add_argument("--config", type=str, default="configs/data/default.yaml", help="Path to data config YAML")
    parser.add_argument("--num_pairs", type=int, default=100, help="Number of counterfactual pairs to generate")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = config.get("data_dir", "./data")

    generator = SyntheticSpatialDatasetGenerator(
        output_dir=output_dir,
        seed=config.get("seed", 42)
    )

    logger.info(f"Generating {args.num_pairs} synthetic counterfactual scene pairs...")
    pairs = generator.generate_dataset(num_pairs=args.num_pairs)
    logger.info(f"Dataset preparation complete! Exported metadata to {output_dir}/metadata/synthetic_metadata.json")


if __name__ == "__main__":
    main()
