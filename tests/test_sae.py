import unittest
from spatial_sae.sae.model import SparseAutoencoder
from spatial_sae.sae.loss import SAELoss
from spatial_sae.sae.metrics import compute_sae_metrics

try:
    import torch
except ImportError:
    torch = None


class TestSAE(unittest.TestCase):
    def test_sae_dimensions_and_loss(self):
        if torch is None:
            self.skipTest("PyTorch is not installed")

        input_dim = 128
        latent_dim = 512
        batch_size = 16

        sae = SparseAutoencoder(input_dim=input_dim, latent_dim=latent_dim, activation_function="relu")
        x = torch.randn(batch_size, input_dim)

        x_hat, z = sae(x)

        self.assertEqual(x_hat.shape, (batch_size, input_dim))
        self.assertEqual(z.shape, (batch_size, latent_dim))

        loss_fn = SAELoss(sparsity_coefficient=0.01)
        total_loss, metrics = loss_fn(x, x_hat, z)

        self.assertGreater(total_loss, 0)
        self.assertIn("rec_loss", metrics)
        self.assertIn("l1_loss", metrics)
        self.assertIn("explained_variance", metrics)

    def test_sae_topk_activation(self):
        if torch is None:
            self.skipTest("PyTorch is not installed")

        input_dim = 64
        latent_dim = 256
        top_k = 16

        sae = SparseAutoencoder(input_dim=input_dim, latent_dim=latent_dim, activation_function="topk", top_k=top_k)
        x = torch.randn(8, input_dim)

        z = sae.encode(x)
        non_zero_count = (z > 0).sum(dim=-1)

        self.assertTrue(torch.all(non_zero_count <= top_k))

    def test_sae_metrics_calculation(self):
        if torch is None:
            self.skipTest("PyTorch is not installed")

        x = torch.randn(10, 32)
        x_hat = x + 0.1 * torch.randn(10, 32)
        z = torch.relu(torch.randn(10, 128))

        metrics = compute_sae_metrics(x, x_hat, z)
        self.assertIn("rec_mse", metrics)
        self.assertIn("explained_variance", metrics)
        self.assertIn("l0_norm", metrics)
        self.assertIn("dead_features_pct", metrics)


if __name__ == "__main__":
    unittest.main()
