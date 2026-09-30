import unittest
from spatial_sae.models.vlm import MockVLMAdapter
from spatial_sae.sae.model import SparseAutoencoder
from spatial_sae.intervention.sae_intervention import SAEInterventionEngine
from spatial_sae.intervention.strategies import AddStrategy, ClampStrategy, SuppressStrategy, ScaleStrategy

try:
    import torch
except ImportError:
    torch = None


class TestIntervention(unittest.TestCase):
    def test_intervention_strategies(self):
        if torch is None:
            self.skipTest("PyTorch is not installed")

        z = torch.ones(4, 10) * 2.0

        # Add Strategy
        add_strat = AddStrategy(feature_indices=[1, 3], alpha=5.0)
        z_add = add_strat.apply(z)
        self.assertEqual(z_add[0, 1].item(), 7.0)
        self.assertEqual(z_add[0, 0].item(), 2.0)

        # Clamp Strategy
        clamp_strat = ClampStrategy(feature_indices=[2], clamp_value=10.0)
        z_clamp = clamp_strat.apply(z)
        self.assertEqual(z_clamp[0, 2].item(), 10.0)

        # Suppress Strategy
        supp_strat = SuppressStrategy(feature_indices=[4])
        z_supp = supp_strat.apply(z)
        self.assertEqual(z_supp[0, 4].item(), 0.0)

        # Scale Strategy
        scale_strat = ScaleStrategy(feature_indices=[5], beta=3.0)
        z_scale = scale_strat.apply(z)
        self.assertEqual(z_scale[0, 5].item(), 6.0)

    def test_sae_intervention_engine(self):
        if torch is None:
            self.skipTest("PyTorch is not installed")

        input_dim = 64
        latent_dim = 256

        vlm = MockVLMAdapter(hidden_dim=input_dim)
        sae = SparseAutoencoder(input_dim=input_dim, latent_dim=latent_dim)

        engine = SAEInterventionEngine(vlm_adapter=vlm, sae=sae, target_layer="model.layers.16")
        strategy = engine.create_strategy(mode="add", feature_indices=[0], strength=2.0)

        h_original = torch.randn(2, 8, input_dim)
        h_modified = engine.transform_activation(h_original, strategy)

        self.assertEqual(h_modified.shape, h_original.shape)


if __name__ == "__main__":
    unittest.main()
