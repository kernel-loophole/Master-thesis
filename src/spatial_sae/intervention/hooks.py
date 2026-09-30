from typing import Any, Callable
from spatial_sae.models.hooks import ForwardHook

try:
    import torch
except ImportError:
    torch = None


class VLMInterventionHook(ForwardHook):
    """Forward hook that intercepts model hidden state, applies SAE intervention fn, and passes reconstructed state back."""

    def __init__(self, name: str, intervention_transform: Callable[[Any], Any]):
        super().__init__(name)
        self.intervention_transform = intervention_transform

    def __call__(self, module: Any, input_tensor: Any, output_tensor: Any) -> Any:
        if isinstance(output_tensor, tuple):
            h = output_tensor[0]
            h_mod = self.intervention_transform(h)
            return (h_mod,) + output_tensor[1:]
        else:
            return self.intervention_transform(output_tensor)
