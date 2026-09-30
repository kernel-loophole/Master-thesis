#!/usr/bin/env python3
"""CLI Script to extract VLM activations across target layers."""

import argparse
from pathlib import Path

from spatial_sae.utils.config import load_config
from spatial_sae.utils.logging import setup_logging
from spatial_sae.models.vlm import HuggingFaceVLMAdapter, MockVLMAdapter
from spatial_sae.data.datasets import SpatialReasoningDataset
from spatial_sae.extraction.activations import ActivationExtractor

logger = setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Extract VLM Activations")
    parser.add_argument("--config", type=str, default="configs/extraction/default.yaml", help="Extraction config path")
    parser.add_argument("--model_config", type=str, default="configs/model/default.yaml", help="Model config path")
    parser.add_argument("--dry_run", action="store_true", help="Use MockVLMAdapter for testing")
    args = parser.parse_args()

    ext_config = load_config(args.config)
    model_config = load_config(args.model_config)

    if args.dry_run or model_config.get("adapter_type") == "mock":
        vlm = MockVLMAdapter(hidden_dim=ext_config.get("hidden_dim", 512))
        vlm.load_model("mock")
    else:
        vlm = HuggingFaceVLMAdapter()
        vlm.load_model(model_config.model_name, device=model_config.device, dtype=model_config.dtype)

    dataset_path = Path("data/metadata/synthetic_metadata.json")
    if not dataset_path.exists():
        logger.warning(f"Metadata file {dataset_path} not found. Generate data first using scripts/prepare_data.py")
        return

    dataset = SpatialReasoningDataset(dataset_path)

    extractor = ActivationExtractor(
        vlm_adapter=vlm,
        target_layers=ext_config.target_layers,
        storage_dir=ext_config.storage_dir,
        batch_size=ext_config.batch_size
    )

    logger.info("Extracting activations...")
    shards = extractor.extract_dataset(dataset, max_samples=ext_config.get("max_samples", 100))
    logger.info(f"Activation extraction complete! Saved {len(shards)} shards to {ext_config.storage_dir}")


if __name__ == "__main__":
    main()
