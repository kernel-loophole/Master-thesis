from typing import Dict, Any

try:
    import torch
except ImportError:
    torch = None


def compute_sae_metrics(
    x: Any,
    x_hat: Any,
    z: Any,
    dead_feature_threshold: float = 1e-6
) -> Dict[str, float]:
    """Compute detailed evaluation metrics for SAE quality and feature sparsity."""
    if torch is None or not isinstance(x, torch.Tensor):
        return {
            "rec_mse": 0.0,
            "explained_variance": 0.0,
            "l0_norm": 0.0,
            "l1_norm": 0.0,
            "dead_features_count": 0,
            "dead_features_pct": 0.0,
        }

    rec_mse = torch.mean((x - x_hat) ** 2).item()
    total_var = torch.var(x).item()
    res_var = torch.var(x - x_hat).item()
    explained_variance = 1.0 - (res_var / (total_var + 1e-8))

    l0_norm = torch.mean((torch.abs(z) > 1e-5).float().sum(dim=-1)).item()
    l1_norm = torch.mean(torch.sum(torch.abs(z), dim=-1)).item()

    # Feature activation frequencies across batch
    feature_max_act = torch.max(z, dim=0).values
    dead_features = (feature_max_act < dead_feature_threshold).sum().item()
    total_features = z.shape[-1]
    dead_features_pct = (dead_features / total_features) * 100.0

    return {
        "rec_mse": rec_mse,
        "explained_variance": explained_variance,
        "l0_norm": l0_norm,
        "l1_norm": l1_norm,
        "dead_features_count": int(dead_features),
        "dead_features_pct": dead_features_pct,
    }
