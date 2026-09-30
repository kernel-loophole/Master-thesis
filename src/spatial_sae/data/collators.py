from typing import List, Dict, Any

try:
    import torch
except ImportError:
    torch = None


class SpatialDataCollator:
    """Collates SpatialSample dictionaries into batch inputs for VLM inference."""

    def __init__(self, processor: Any = None):
        self.processor = processor

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        sample_ids = [b["sample_id"] for b in batch]
        questions = [b["question"] for b in batch]
        answers = [b["answer"] for b in batch]
        spatial_relations = [b["spatial_relation"] for b in batch]
        attributes = [b["attributes"] for b in batch]
        images = [b["image"] for b in batch if b["image"] is not None]

        collated = {
            "sample_ids": sample_ids,
            "questions": questions,
            "answers": answers,
            "spatial_relations": spatial_relations,
            "attributes": attributes,
            "images": images,
        }

        if self.processor and images and torch is not None:
            processed_inputs = self.processor(
                text=questions,
                images=images,
                return_tensors="pt",
                padding=True
            )
            collated["input_ids"] = processed_inputs.get("input_ids")
            collated["pixel_values"] = processed_inputs.get("pixel_values")

        return collated


class ActivationCollator:
    """Collates activation tensors for SAE training."""

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        activations = [b["activation"] for b in batch]
        metadatas = [b["metadata"] for b in batch]

        if torch is not None and isinstance(activations[0], torch.Tensor):
            stacked_activations = torch.stack(activations, dim=0)
        else:
            stacked_activations = activations

        return {
            "activations": stacked_activations,
            "metadata": metadatas,
        }
