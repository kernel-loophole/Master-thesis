import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    # Dry run fallback interface if PyTorch is absent
    class Dataset:
        pass


class SpatialReasoningDataset(Dataset):
    """PyTorch Dataset loading spatial samples from JSON metadata."""

    def __init__(self, metadata_path: Union[str, Path], transform=None):
        self.metadata_path = Path(metadata_path)
        self.transform = transform
        self.samples: List[Dict[str, Any]] = []

        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if "original" in item and "counterfactual" in item:
                            self.samples.append(item["original"])
                            self.samples.append(item["counterfactual"])
                        else:
                            self.samples.append(item)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        image_path = item.get("image_path")
        image = None
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)

        return {
            "sample_id": item["sample_id"],
            "image": image,
            "image_path": image_path,
            "question": item["question"],
            "answer": item["answer"],
            "spatial_relation": item["spatial_relation"],
            "attributes": item["attributes"],
            "metadata": item.get("metadata", {}),
        }


class SyntheticSpatialDataset(SpatialReasoningDataset):
    """Wrapper for synthetic spatial reasoning datasets."""
    pass


class ActivationDataset(Dataset):
    """PyTorch Dataset loading stored activation tensors for SAE training."""

    def __init__(self, activations: Any, metadata: Optional[List[Dict[str, Any]]] = None):
        """
        Args:
            activations: torch.Tensor of shape [N, activation_dim] or list of tensors
            metadata: Optional list of metadata dicts corresponding to each activation
        """
        self.activations = activations
        self.metadata = metadata or []

    def __len__(self) -> int:
        return len(self.activations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        act = self.activations[idx]
        meta = self.metadata[idx] if idx < len(self.metadata) else {}
        return {
            "activation": act,
            "metadata": meta,
        }
