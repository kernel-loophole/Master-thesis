from typing import Dict, Any, Callable, Optional, List

try:
    import torch
    import torch.nn as nn
except ImportError:
    nn = None
    torch = None


class ForwardHook:
    """Base class for PyTorch forward hooks."""

    def __init__(self, name: str):
        self.name = name
        self.handle = None

    def register(self, module: Any) -> None:
        if hasattr(module, "register_forward_hook"):
            self.handle = module.register_forward_hook(self)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def __call__(self, module: Any, input_tensor: Any, output_tensor: Any) -> Any:
        raise NotImplementedError


class ActivationCollectorHook(ForwardHook):
    """Forward hook that captures layer output activations."""

    def __init__(self, name: str, cache_dict: Dict[str, Any]):
        super().__init__(name)
        self.cache_dict = cache_dict

    def __call__(self, module: Any, input_tensor: Any, output_tensor: Any) -> Any:
        # If output is a tuple (e.g. residual stream + attention weights), grab hidden states
        if isinstance(output_tensor, tuple):
            tensor = output_tensor[0]
        else:
            tensor = output_tensor

        if torch is not None and isinstance(tensor, torch.Tensor):
            self.cache_dict[self.name] = tensor.detach()
        else:
            self.cache_dict[self.name] = tensor

        return output_tensor


class SteeringInterventionHook(ForwardHook):
    """Forward hook that mutates layer output activations during forward passes."""

    def __init__(self, name: str, intervention_fn: Callable[[Any], Any]):
        super().__init__(name)
        self.intervention_fn = intervention_fn

    def __call__(self, module: Any, input_tensor: Any, output_tensor: Any) -> Any:
        if isinstance(output_tensor, tuple):
            modified = self.intervention_fn(output_tensor[0])
            return (modified,) + output_tensor[1:]
        else:
            return self.intervention_fn(output_tensor)


class HookManager:
    """Manages registration and cleanup of multiple hooks across VLM layers."""

    def __init__(self):
        self.hooks: List[ForwardHook] = []

    def register_collector(self, name: str, module: Any, cache_dict: Dict[str, Any]) -> ActivationCollectorHook:
        hook = ActivationCollectorHook(name, cache_dict)
        hook.register(module)
        self.hooks.append(hook)
        return hook

    def register_steering(self, name: str, module: Any, intervention_fn: Callable[[Any], Any]) -> SteeringInterventionHook:
        hook = SteeringInterventionHook(name, intervention_fn)
        hook.register(module)
        self.hooks.append(hook)
        return hook

    def remove_all(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
