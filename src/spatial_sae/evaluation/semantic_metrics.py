from typing import Dict, Any, List

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


class SemanticPreservationEvaluator:
    """Evaluates semantic distribution shifts and KL divergence under steering interventions."""

    def __init__(self):
        pass

    def compute_kl_divergence(self, clean_logits: Any, steered_logits: Any) -> float:
        """Compute KL divergence KL(P_clean || P_steered) over vocabulary logits."""
        if torch is None or not isinstance(clean_logits, torch.Tensor):
            return 0.0

        p_clean = F.softmax(clean_logits.float(), dim=-1)
        log_p_steered = F.log_softmax(steered_logits.float(), dim=-1)

        kl_div = F.kl_div(log_p_steered, p_clean, reduction="batchmean")
        return float(kl_div.item())

    def compute_perplexity_shift(self, clean_loss: float, steered_loss: float) -> float:
        """Compute relative perplexity degradation ratio."""
        return max(0.0, steered_loss - clean_loss)
