import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union


def save_checkpoint(
    state_dict: Dict[str, Any],
    metadata: Dict[str, Any],
    filepath: Union[str, Path]
) -> None:
    """Save PyTorch checkpoint state dict alongside metadata JSON atomically."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        temp_pt = filepath.with_suffix(".tmp")
        torch.save(state_dict, temp_pt)
        os.replace(temp_pt, filepath)
    except ImportError:
        # Fallback if torch is not installed during dry run
        pass

    metadata_path = filepath.with_suffix(".json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)


def load_checkpoint(filepath: Union[str, Path], device: str = "cpu") -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Load checkpoint weights and metadata JSON."""
    filepath = Path(filepath)
    metadata_path = filepath.with_suffix(".json")

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    try:
        import torch
        state_dict = torch.load(filepath, map_location=device)
    except ImportError:
        state_dict = {}

    return state_dict, metadata
