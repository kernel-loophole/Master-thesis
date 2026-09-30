import unittest
from spatial_sae.models.vlm import MockVLMAdapter


class TestVLM(unittest.TestCase):
    def test_mock_vlm_adapter(self):
        vlm = MockVLMAdapter(hidden_dim=256)
        vlm.load_model("mock", device="cpu")

        inputs = vlm.prepare_inputs([], ["What is the relation?"])
        outputs = vlm.forward(inputs)

        self.assertIsNotNone(outputs)

        act = vlm.get_activation("model.layers.16", inputs)
        try:
            import torch
            if isinstance(act, torch.Tensor):
                self.assertEqual(act.ndim, 3)
                self.assertEqual(act.shape[-1], 256)
        except ImportError:
            pass


if __name__ == "__main__":
    unittest.main()
