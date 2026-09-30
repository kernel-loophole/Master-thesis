from typing import Any, List, Optional
from spatial_sae.sae.model import SparseAutoencoder
from spatial_sae.models.vlm import VLMAdapter
from spatial_sae.intervention.strategies import (
    InterventionStrategy,
    AddStrategy,
    ClampStrategy,
    SuppressStrategy,
    ScaleStrategy
)
from spatial_sae.intervention.hooks import VLMInterventionHook

try:
    import torch
except ImportError:
    torch = None


class SAEInterventionEngine:
    """Core intervention engine that modifies model activations in SAE latent space."""

    def __init__(
        self,
        vlm_adapter: VLMAdapter,
        sae: SparseAutoencoder,
        target_layer: str
    ):
        self.vlm_adapter = vlm_adapter
        self.sae = sae
        self.target_layer = target_layer
        self.active_hook = None

    def transform_activation(self, h: Any, strategy: InterventionStrategy) -> Any:
        """
        Flow:
        1. h -> z = Encoder(h)
        2. z -> z' (modify selected feature via strategy)
        3. z' -> h' = Decoder(z')
        """
        if torch is None or not isinstance(h, torch.Tensor):
            return h

        device = next(self.sae.parameters()).device
        h_device = h.to(device)

        # 1. Encode activation into SAE latent space
        z = self.sae.encode(h_device)

        # 2. Modify selected feature(s)
        z_mod = strategy.apply(z)

        # 3. Reconstruct modified activation vector
        h_mod = self.sae.decode(z_mod)

        # Return modified hidden state in original tensor dtype and device
        return h_mod.to(h.dtype).to(h.device)

    def register_intervention(self, strategy: InterventionStrategy) -> VLMInterventionHook:
        """Register forward hook on VLM target layer with active intervention strategy."""
        transform_fn = lambda h: self.transform_activation(h, strategy)
        hook = self.vlm_adapter.register_hook(self.target_layer, transform_fn)
        self.active_hook = hook
        return hook

    def remove_intervention(self) -> None:
        """Remove registered intervention hook from VLM."""
        if self.vlm_adapter:
            self.vlm_adapter.remove_hooks()
        self.active_hook = None

    def create_strategy(
        self,
        mode: str,
        feature_indices: List[int],
        strength: float = 5.0,
        clamp_val: float = 10.0,
        scale_beta: float = 2.0
    ) -> InterventionStrategy:
        """Factory method for instantiating intervention strategy by name."""
        mode = mode.lower()
        if mode == "add":
            return AddStrategy(feature_indices, alpha=strength)
        elif mode == "clamp":
            return ClampStrategy(feature_indices, clamp_value=clamp_val)
        elif mode == "suppress":
            return SuppressStrategy(feature_indices)
        elif mode == "scale":
            return ScaleStrategy(feature_indices, beta=scale_beta)
        else:
            raise ValueError(f"Unsupported intervention mode: {mode}. Expected 'add', 'clamp', 'suppress', or 'scale'.")
