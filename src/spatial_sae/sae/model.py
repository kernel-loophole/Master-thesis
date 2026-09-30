import math
from typing import Tuple, Dict, Any, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    nn = type("nn", (), {"Module": object})
    torch = None
    F = None


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder (SAE) for learning interpretable latent feature bases from model activations."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        activation_function: str = "relu",
        top_k: int = 32,
        normalize_decoder: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation_function = activation_function.lower()
        self.top_k = top_k
        self.normalize_decoder = normalize_decoder

        if torch is not None:
            # Encoder & Decoder weights and biases
            self.W_enc = nn.Parameter(torch.empty(input_dim, latent_dim))
            self.b_enc = nn.Parameter(torch.zeros(latent_dim))

            self.W_dec = nn.Parameter(torch.empty(latent_dim, input_dim))
            self.b_dec = nn.Parameter(torch.zeros(input_dim))

            self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights following standard SAE initialization practices."""
        if torch is None:
            return
        # Kaiming uniform initialization for encoder
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        # Decoder initialization matched to transpose of encoder and normalized
        with torch.no_grad():
            self.W_dec.data = self.W_enc.data.t().clone()
            if self.normalize_decoder:
                self.normalize_decoder_weights()

    def normalize_decoder_weights(self) -> None:
        """Enforce unit norm constraints on decoder feature vectors: ||W_dec_i||_2 = 1."""
        if torch is None or not hasattr(self, "W_dec"):
            return
        with torch.no_grad():
            norms = torch.norm(self.W_dec.data, dim=1, keepdim=True)
            self.W_dec.data.div_(norms + 1e-8)

    def encode(self, x: Any) -> Any:
        """Encode input activation x into sparse latent feature activations z."""
        if torch is None:
            return x

        # Pre-subtract decoder bias
        x_centered = x - self.b_dec
        pre_act = torch.matmul(x_centered, self.W_enc) + self.b_enc

        if self.activation_function == "relu":
            z = F.relu(pre_act)
        elif self.activation_function == "topk":
            # Top-K sparsity selection
            topk_vals, topk_indices = torch.topk(pre_act, k=self.top_k, dim=-1)
            z = torch.zeros_like(pre_act)
            z.scatter_(-1, topk_indices, F.relu(topk_vals))
        else:
            z = F.relu(pre_act)

        return z

    def decode(self, z: Any) -> Any:
        """Reconstruct input activation x_hat from sparse latent z."""
        if torch is None:
            return z
        x_hat = torch.matmul(z, self.W_dec) + self.b_dec
        return x_hat

    def forward(self, x: Any) -> Tuple[Any, Any]:
        """Forward pass: x -> encode -> z -> decode -> x_hat."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
