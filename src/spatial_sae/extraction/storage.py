import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

try:
    import torch
except ImportError:
    torch = None

try:
    from safetensors.torch import save_file as save_safetensors, load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


class ActivationStorage:
    """Manages disk storage and chunking of activation tensors and sample metadata."""

    def __init__(self, storage_dir: Union[str, Path]):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_shard(
        self,
        activations: Dict[str, Any],  # layer_name -> torch.Tensor
        metadata: List[Dict[str, Any]],
        shard_id: int
    ) -> Path:
        """Save a single shard of activation tensors and metadata."""
        shard_dir = self.storage_dir / f"shard_{shard_id:04d}"
        shard_dir.mkdir(parents=True, exist_ok=True)

        if torch is not None:
            if HAS_SAFETENSORS:
                # Convert keys for safetensors compliance (replace dot with underscore in layer names)
                st_dict = {k.replace(".", "_"): v.contiguous() for k, v in activations.items() if isinstance(v, torch.Tensor)}
                save_safetensors(st_dict, str(shard_dir / "activations.safetensors"))
            else:
                torch.save(activations, shard_dir / "activations.pt")

        meta_path = shard_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        return shard_dir

    def load_shard(self, shard_id: int) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Load activation tensors and metadata from a specific shard."""
        shard_dir = self.storage_dir / f"shard_{shard_id:04d}"
        if not shard_dir.exists():
            raise FileNotFoundError(f"Shard directory not found: {shard_dir}")

        meta_path = shard_dir / "metadata.json"
        metadata = []
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        activations = {}
        st_path = shard_dir / "activations.safetensors"
        pt_path = shard_dir / "activations.pt"

        if torch is not None:
            if st_path.exists() and HAS_SAFETENSORS:
                st_dict = load_safetensors(str(st_path))
                activations = st_dict
            elif pt_path.exists():
                activations = torch.load(pt_path, map_location="cpu")

        return activations, metadata

    def get_all_shards(self) -> List[int]:
        """List all available shard indices in storage directory."""
        shards = []
        for d in self.storage_dir.glob("shard_*"):
            if d.is_dir():
                try:
                    shard_id = int(d.name.split("_")[1])
                    shards.append(shard_id)
                except ValueError:
                    pass
        return sorted(shards)
