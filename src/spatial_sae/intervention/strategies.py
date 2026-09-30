from abc import ABC, abstractmethod
from typing import List, Any

try:
    import torch
except ImportError:
    torch = None


class InterventionStrategy(ABC):
    """Abstract base class for SAE feature intervention strategies."""

    def __init__(self, feature_indices: List[int]):
        self.feature_indices = feature_indices

    @abstractmethod
    def apply(self, z: Any) -> Any:
        """Modify latent feature tensor z in-place or return modified tensor."""
        pass


class AddStrategy(InterventionStrategy):
    """Add constant alpha to selected SAE features: z'_i = z_i + alpha."""

    def __init__(self, feature_indices: List[int], alpha: float = 5.0):
        super().__init__(feature_indices)
        self.alpha = alpha

    def apply(self, z: Any) -> Any:
        if torch is None or not isinstance(z, torch.Tensor):
            return z
        z_mod = z.clone()
        for idx in self.feature_indices:
            z_mod[..., idx] += self.alpha
        return z_mod


class ClampStrategy(InterventionStrategy):
    """Clamp selected SAE features to fixed value c: z'_i = c."""

    def __init__(self, feature_indices: List[int], clamp_value: float = 10.0):
        super().__init__(feature_indices)
        self.clamp_value = clamp_value

    def apply(self, z: Any) -> Any:
        if torch is None or not isinstance(z, torch.Tensor):
            return z
        z_mod = z.clone()
        for idx in self.feature_indices:
            z_mod[..., idx] = self.clamp_value
        return z_mod


class SuppressStrategy(InterventionStrategy):
    """Zero-out / ablate selected SAE features: z'_i = 0."""

    def __init__(self, feature_indices: List[int]):
        super().__init__(feature_indices)

    def apply(self, z: Any) -> Any:
        if torch is None or not isinstance(z, torch.Tensor):
            return z
        z_mod = z.clone()
        for idx in self.feature_indices:
            z_mod[..., idx] = 0.0
        return z_mod


class ScaleStrategy(InterventionStrategy):
    """Scale selected SAE features by beta multiplier: z'_i = beta * z_i."""

    def __init__(self, feature_indices: List[int], beta: float = 2.0):
        super().__init__(feature_indices)
        self.beta = beta

    def apply(self, z: Any) -> Any:
        if torch is None or not isinstance(z, torch.Tensor):
            return z
        z_mod = z.clone()
        for idx in self.feature_indices:
            z_mod[..., idx] *= self.beta
        return z_mod
