from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Tuple
import os

from spatial_sae.models.hooks import HookManager

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


class VLMAdapter(ABC):
    """Abstract interface for Vision-Language Models (VLM) adapters."""

    @abstractmethod
    def load_model(self, model_name: str, device: str = "cuda", dtype: str = "bfloat16") -> None:
        """Load model weights and processor."""
        pass

    @abstractmethod
    def prepare_inputs(self, images: List[Any], text_prompts: List[str]) -> Dict[str, Any]:
        """Preprocess text and images into model input tensors."""
        pass

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Any:
        """Run standard forward pass."""
        pass

    @abstractmethod
    def get_activation(self, layer_name: str, inputs: Dict[str, Any]) -> Any:
        """Extract activation tensor for a specific layer."""
        pass

    @abstractmethod
    def register_hook(self, layer_name: str, hook_fn: Callable) -> Any:
        """Register PyTorch forward hook on target layer."""
        pass

    @abstractmethod
    def remove_hooks(self) -> None:
        """Remove all active hooks."""
        pass


class HuggingFaceVLMAdapter(VLMAdapter):
    """Concrete VLM adapter for Hugging Face Transformers vision-language models."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda"
        self.dtype = "bfloat16"
        self.hook_manager = HookManager()
        self.activation_cache: Dict[str, Any] = {}

    def load_model(self, model_name: str, device: str = "cuda", dtype: str = "bfloat16") -> None:
        self.device = device
        self.dtype = dtype

        if torch is None:
            raise RuntimeError("PyTorch is required to load HuggingFace VLM models.")

        from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM

        torch_dtype = getattr(torch, dtype, torch.bfloat16)

        try:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )
        except Exception:
            # Fallback to AutoModelForCausalLM if Vision2Seq doesn't match model type
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )

        self.model.eval()

    def _get_module_by_name(self, layer_name: str) -> Any:
        """Resolve nested layer module by dot-separated string name."""
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        curr = self.model
        for token in layer_name.split("."):
            if token.isdigit():
                curr = curr[int(token)]
            else:
                curr = getattr(curr, token)
        return curr

    def prepare_inputs(self, images: List[Any], text_prompts: List[str]) -> Dict[str, Any]:
        if self.processor is None:
            raise RuntimeError("Processor is not loaded.")
        inputs = self.processor(text=text_prompts, images=images, return_tensors="pt", padding=True)
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    def forward(self, inputs: Dict[str, Any]) -> Any:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        with torch.no_grad():
            return self.model(**inputs)

    def get_activation(self, layer_name: str, inputs: Dict[str, Any]) -> Any:
        module = self._get_module_by_name(layer_name)
        self.activation_cache.clear()
        self.hook_manager.register_collector(layer_name, module, self.activation_cache)
        _ = self.forward(inputs)
        act = self.activation_cache.get(layer_name)
        self.hook_manager.remove_all()
        return act

    def register_hook(self, layer_name: str, hook_fn: Callable) -> Any:
        module = self._get_module_by_name(layer_name)
        return self.hook_manager.register_steering(layer_name, module, hook_fn)

    def remove_hooks(self) -> None:
        self.hook_manager.remove_all()


class MockVLMAdapter(VLMAdapter):
    """Lightweight mock VLM adapter for fast testing without loading multi-GB weights."""

    def __init__(self, hidden_dim: int = 512):
        self.hidden_dim = hidden_dim
        self.loaded = False
        self.hooks = {}
        self.device = "cpu"

    def load_model(self, model_name: str, device: str = "cpu", dtype: str = "float32") -> None:
        self.loaded = True
        self.device = device

    def _get_module_by_name(self, layer_name: str) -> Any:
        """Mock module returning self for forward hook compatibility."""
        return self

    def prepare_inputs(self, images: List[Any], text_prompts: List[str]) -> Dict[str, Any]:
        batch_size = max(len(text_prompts), len(images) if images else 1)
        if torch is not None:
            dummy_ids = torch.ones((batch_size, 16), dtype=torch.long)
            dummy_pixel = torch.randn((batch_size, 3, 224, 224))
            return {"input_ids": dummy_ids, "pixel_values": dummy_pixel}
        return {"batch_size": batch_size}

    def forward(self, inputs: Dict[str, Any]) -> Any:
        batch_size = inputs.get("input_ids", torch.ones((1, 16))).shape[0] if torch is not None and isinstance(inputs.get("input_ids"), torch.Tensor) else 1
        if torch is not None:
            return type("MockOutput", (), {
                "logits": torch.randn((batch_size, 16, 32000)),
                "hidden_states": torch.randn((batch_size, 16, self.hidden_dim))
            })()
        return {"logits": "mock_logits"}

    def get_activation(self, layer_name: str, inputs: Dict[str, Any]) -> Any:
        batch_size = inputs.get("input_ids", torch.ones((1, 16))).shape[0] if torch is not None and isinstance(inputs.get("input_ids"), torch.Tensor) else 1
        if torch is not None:
            act = torch.randn((batch_size, 16, self.hidden_dim))
            if layer_name in self.hooks:
                act = self.hooks[layer_name](act)
            return act
        return None

    def register_hook(self, layer_name: str, hook_fn: Callable) -> Any:
        self.hooks[layer_name] = hook_fn
        return hook_fn

    def remove_hooks(self) -> None:
        self.hooks.clear()
