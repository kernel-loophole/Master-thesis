from typing import Dict, Any, List, Optional
from spatial_sae.models.hooks import HookManager

try:
    import torch
except ImportError:
    torch = None


class LayerActivationCollector:
    """Manages layer-wise activation collection across multiple VLM modules."""

    def __init__(self, vlm_adapter: Any, target_layers: List[str]):
        self.vlm_adapter = vlm_adapter
        self.target_layers = target_layers
        self.cache: Dict[str, Any] = {}
        self.hook_manager = HookManager()

    def register_collectors(self) -> None:
        self.cache.clear()
        for layer_name in self.target_layers:
            module = self.vlm_adapter._get_module_by_name(layer_name)
            self.hook_manager.register_collector(layer_name, module, self.cache)

    def remove_collectors(self) -> None:
        self.hook_manager.remove_all()

    def collect(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run forward pass and return captured layer activations."""
        self.register_collectors()
        _ = self.vlm_adapter.forward(inputs)
        results = {k: v for k, v in self.cache.items()}
        self.remove_collectors()
        return results
