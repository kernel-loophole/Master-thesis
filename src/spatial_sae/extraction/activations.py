from pathlib import Path
from typing import Dict, Any, List, Optional, Union

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda iterable, **kwargs: iterable

from spatial_sae.models.vlm import VLMAdapter
from spatial_sae.extraction.collectors import LayerActivationCollector
from spatial_sae.extraction.storage import ActivationStorage
from spatial_sae.data.schemas import ExtractionMetadata
from spatial_sae.utils.logging import get_logger

try:
    import torch
except ImportError:
    torch = None

logger = get_logger("spatial_sae.extraction")


class ActivationExtractor:
    """Reusable pipeline for extracting, collecting, and storing VLM activations."""

    def __init__(
        self,
        vlm_adapter: VLMAdapter,
        target_layers: List[str],
        storage_dir: Union[str, Path],
        batch_size: int = 8,
        token_position_strategy: str = "last_multimodal_token",
        offload_to_cpu: bool = True
    ):
        self.vlm_adapter = vlm_adapter
        self.target_layers = target_layers
        self.storage = ActivationStorage(storage_dir)
        self.batch_size = batch_size
        self.token_strategy = token_position_strategy
        self.offload_to_cpu = offload_to_cpu
        self.collector = LayerActivationCollector(vlm_adapter, target_layers)

    def extract_dataset(
        self,
        dataset: Any,
        shard_size: int = 500,
        max_samples: Optional[int] = None
    ) -> List[Path]:
        """Run extraction across dataset samples and save shards to disk."""
        logger.info(f"Starting activation extraction for {len(dataset)} samples across layers: {self.target_layers}")

        saved_shard_paths = []
        current_activations: Dict[str, List[Any]] = {layer: [] for layer in self.target_layers}
        current_metadata: List[Dict[str, Any]] = []
        shard_count = 0

        total_count = min(len(dataset), max_samples) if max_samples else len(dataset)

        for i in tqdm(range(0, total_count, self.batch_size), desc="Extracting Activations"):
            batch_items = [dataset[j] for j in range(i, min(i + self.batch_size, total_count))]

            images = [item.get("image") for item in batch_items if item.get("image") is not None]
            questions = [item["question"] for item in batch_items]

            inputs = self.vlm_adapter.prepare_inputs(images, questions)
            layer_acts = self.collector.collect(inputs)

            for idx, item in enumerate(batch_items):
                meta = ExtractionMetadata(
                    sample_id=item["sample_id"],
                    image_id=item.get("image_path"),
                    question=item["question"],
                    spatial_relation=item["spatial_relation"],
                    object_1_color=item["attributes"].get("object_1_color", "") if isinstance(item["attributes"], dict) else getattr(item["attributes"], "object_1_color", ""),
                    object_1_shape=item["attributes"].get("object_1_shape", "") if isinstance(item["attributes"], dict) else getattr(item["attributes"], "object_1_shape", ""),
                    object_2_color=item["attributes"].get("object_2_color", "") if isinstance(item["attributes"], dict) else getattr(item["attributes"], "object_2_color", ""),
                    object_2_shape=item["attributes"].get("object_2_shape", "") if isinstance(item["attributes"], dict) else getattr(item["attributes"], "object_2_shape", ""),
                    layer="all",
                    token_position=-1
                ).to_dict()
                current_metadata.append(meta)

            for layer_name, tensor in layer_acts.items():
                if torch is not None and isinstance(tensor, torch.Tensor):
                    # Slice token according to strategy
                    if self.token_strategy == "last_multimodal_token" and tensor.ndim == 3:
                        tensor = tensor[:, -1, :]  # Select last token activation: [batch, hidden_dim]
                    if self.offload_to_cpu:
                        tensor = tensor.cpu()
                current_activations[layer_name].append(tensor)

            # Check if shard threshold is reached
            if len(current_metadata) >= shard_size or (i + self.batch_size) >= total_count:
                stacked_acts = {}
                for layer_name, act_list in current_activations.items():
                    if torch is not None and len(act_list) > 0 and isinstance(act_list[0], torch.Tensor):
                        stacked_acts[layer_name] = torch.cat(act_list, dim=0)
                    else:
                        stacked_acts[layer_name] = act_list

                shard_path = self.storage.save_shard(stacked_acts, current_metadata, shard_count)
                saved_shard_paths.append(shard_path)
                logger.info(f"Saved shard {shard_count} with {len(current_metadata)} samples to {shard_path}")

                shard_count += 1
                current_activations = {layer: [] for layer in self.target_layers}
                current_metadata = []

        return saved_shard_paths
