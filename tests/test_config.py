import unittest
from pathlib import Path
from spatial_sae.utils.config import load_config, ConfigDict


class TestConfig(unittest.TestCase):
    def test_config_loading(self):
        config_path = Path("configs/sae/default.yaml")
        self.assertTrue(config_path.exists(), "Config file must exist")

        cfg = load_config(config_path)
        self.assertIsInstance(cfg, ConfigDict)
        self.assertTrue(hasattr(cfg, "input_dim"))
        self.assertTrue(hasattr(cfg, "latent_dim"))
        self.assertGreater(cfg.input_dim, 0)
        self.assertGreaterEqual(cfg.sparsity_coefficient, 0.0)

    def test_config_dict_dot_access(self):
        d = ConfigDict({"a": {"b": 42}})
        self.assertEqual(d.a.b, 42)
        d.a.b = 100
        self.assertEqual(d.a.b, 100)


if __name__ == "__main__":
    unittest.main()
