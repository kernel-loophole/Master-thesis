import unittest
import tempfile
from pathlib import Path
from spatial_sae.data.generation import SyntheticSpatialDatasetGenerator
from spatial_sae.data.schemas import SceneAttributes, SpatialSample


class TestDataset(unittest.TestCase):
    def test_synthetic_scene_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = SyntheticSpatialDatasetGenerator(output_dir=tmpdir, seed=42)
            pair = gen.generate_counterfactual_pair(0)

            self.assertEqual(pair.pair_id, "pair_00000")
            self.assertEqual(pair.swapped_attribute, "colors")
            self.assertEqual(pair.sample_original.spatial_relation, pair.sample_counterfactual.spatial_relation)

            # Image paths should be created
            self.assertTrue(Path(pair.sample_original.image_path).exists())
            self.assertTrue(Path(pair.sample_counterfactual.image_path).exists())

    def test_dataset_generation_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = SyntheticSpatialDatasetGenerator(output_dir=tmpdir, seed=42)
            pairs = gen.generate_dataset(num_pairs=5)
            self.assertEqual(len(pairs), 5)

            meta_file = Path(tmpdir) / "metadata" / "synthetic_metadata.json"
            self.assertTrue(meta_file.exists())


if __name__ == "__main__":
    unittest.main()
