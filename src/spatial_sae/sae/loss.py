from typing import Tuple, Dict, Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    nn = type("nn", (), {"Module": object})
    torch = None
    F = None


class SAELoss(nn.Module):
    """Computes SAE training objective: L = L_reconstruction + λ * L_sparsity."""

    def __init__(self, sparsity_coefficient: float = 0.001):
        super().__init__()
        self.sparsity_coefficient = sparsity_coefficient

    def forward(self, x: Any, x_hat: Any, z: Any) -> Tuple[Any, Dict[str, float]]:
        if torch is None:
            return 0.0, {}

        # Reconstruction Loss (MSE)
        rec_loss = F.mse_loss(x_hat, x, reduction="mean")

        # Normalized MSE: MSE / Var(x)
        x_variance = torch.var(x, dim=0).sum()
        norm_rec_loss = rec_loss / (x_variance + 1e-8)

        # Sparsity Loss (L1 norm of latent feature activations)
        l1_loss = torch.norm(z, p=1, dim=-1).mean()

        total_loss = rec_loss + self.sparsity_coefficient * l1_loss

        # Explained Variance: 1 - Var(x - x_hat) / Var(x)
        residual_variance = torch.var(x - x_hat, dim=0).sum()
        explained_variance = 1.0 - (residual_variance / (x_variance + 1e-8))

        # L0 Norm (mean count of non-zero active features)
        l0_norm = (z > 1e-5).float().sum(dim=-1).mean()

        metrics = {
            "total_loss": total_loss.item(),
            "rec_loss": rec_loss.item(),
            "norm_rec_loss": norm_rec_loss.item(),
            "l1_loss": l1_loss.item(),
            "l0_norm": l0_norm.item(),
            "explained_variance": explained_variance.item(),
        }

        return total_loss, metrics
