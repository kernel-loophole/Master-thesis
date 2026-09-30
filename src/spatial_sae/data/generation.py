import os
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image, ImageDraw

from spatial_sae.data.schemas import SpatialSample, SceneAttributes, CounterfactualPair


COLOR_MAP = {
    "red": (220, 50, 50),
    "blue": (50, 100, 220),
    "green": (50, 180, 50),
    "yellow": (230, 210, 40),
    "purple": (150, 50, 200),
    "cyan": (40, 200, 220),
    "orange": (240, 140, 30),
    "gray": (150, 150, 150),
}


class SyntheticSpatialDatasetGenerator:
    """Generates synthetic 2D spatial reasoning scenes with counterfactual pairs."""

    def __init__(
        self,
        output_dir: str = "./data",
        image_size: Tuple[int, int] = (256, 256),
        seed: int = 42
    ):
        self.output_dir = Path(output_dir)
        self.image_dir = self.output_dir / "raw" / "synthetic_images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size
        self.rng = random.Random(seed)

        self.relations = ["inside", "outside", "left_of", "right_of", "above", "below", "overlapping", "behind"]
        self.colors = list(COLOR_MAP.keys())
        self.shapes = ["ball", "box", "cube", "sphere", "cylinder"]

    def _draw_shape(
        self,
        draw: ImageDraw.ImageDraw,
        shape: str,
        color_rgb: Tuple[int, int, int],
        bbox: Tuple[int, int, int, int]
    ):
        x0, y0, x1, y1 = bbox
        if shape in ["ball", "sphere"]:
            draw.ellipse([x0, y0, x1, y1], fill=color_rgb, outline=(0, 0, 0), width=2)
        elif shape in ["box", "cube"]:
            draw.rectangle([x0, y0, x1, y1], fill=color_rgb, outline=(0, 0, 0), width=2)
        elif shape == "cylinder":
            draw.rounded_rectangle([x0, y0, x1, y1], radius=10, fill=color_rgb, outline=(0, 0, 0), width=2)
        else:
            draw.rectangle([x0, y0, x1, y1], fill=color_rgb, outline=(0, 0, 0), width=2)

    def generate_scene(
        self,
        attributes: SceneAttributes,
        relation: str,
        filename: str
    ) -> Tuple[str, Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """Render a synthetic 2D image based on spatial relation and object attributes."""
        img = Image.new("RGB", self.image_size, color=(245, 245, 245))
        draw = ImageDraw.Draw(img)
        w, h = self.image_size

        # Compute bounding boxes according to spatial relation
        if relation == "inside":
            bbox2 = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)  # Container box
            bbox1 = (w // 3 + 10, h // 3 + 10, 2 * w // 3 - 10, 2 * h // 3 - 10)  # Contained object
        elif relation == "outside":
            bbox2 = (w // 6, h // 3, 2 * w // 5, 2 * h // 3)
            bbox1 = (3 * w // 5, h // 3, 5 * w // 6, 2 * h // 3)
        elif relation == "left_of":
            bbox1 = (w // 8, h // 3, 3 * w // 8, 2 * h // 3)
            bbox2 = (5 * w // 8, h // 3, 7 * w // 8, 2 * h // 3)
        elif relation == "right_of":
            bbox2 = (w // 8, h // 3, 3 * w // 8, 2 * h // 3)
            bbox1 = (5 * w // 8, h // 3, 7 * w // 8, 2 * h // 3)
        elif relation == "above":
            bbox1 = (w // 3, h // 8, 2 * w // 3, 3 * h // 8)
            bbox2 = (w // 3, 5 * h // 8, 2 * w // 3, 7 * h // 8)
        elif relation == "below":
            bbox2 = (w // 3, h // 8, 2 * w // 3, 3 * h // 8)
            bbox1 = (w // 3, 5 * h // 8, 2 * w // 3, 7 * h // 8)
        elif relation == "overlapping":
            bbox1 = (w // 4, h // 3, 3 * w // 5, 2 * h // 3)
            bbox2 = (2 * w // 5, h // 3, 3 * w // 4, 2 * h // 3)
        elif relation == "behind":
            bbox1 = (w // 3 + 15, h // 3 - 15, 2 * w // 3 + 15, 2 * h // 3 - 15)  # Slightly smaller/offset
            bbox2 = (w // 3, h // 3, 2 * w // 3, 2 * h // 3)
        else:
            bbox1 = (w // 8, h // 3, 3 * w // 8, 2 * h // 3)
            bbox2 = (5 * w // 8, h // 3, 7 * w // 8, 2 * h // 3)

        col1 = COLOR_MAP.get(attributes.object_1_color, (100, 100, 100))
        col2 = COLOR_MAP.get(attributes.object_2_color, (100, 100, 100))

        # Render background/container shape first if inside/behind
        if relation in ["inside", "behind"]:
            self._draw_shape(draw, attributes.object_2_shape, col2, bbox2)
            self._draw_shape(draw, attributes.object_1_shape, col1, bbox1)
        else:
            self._draw_shape(draw, attributes.object_1_shape, col1, bbox1)
            self._draw_shape(draw, attributes.object_2_shape, col2, bbox2)

        file_path = str(self.image_dir / filename)
        img.save(file_path)
        return file_path, bbox1, bbox2

    def generate_counterfactual_pair(self, idx: int) -> CounterfactualPair:
        """Generate a pair of scenes with matched spatial layout but swapped object color/shape attributes."""
        rel = self.rng.choice(self.relations)
        color_a, color_b = self.rng.sample(self.colors, 2)
        shape_a, shape_b = self.rng.sample(self.shapes, 2)

        # Original attributes
        attr_orig = SceneAttributes(
            object_1_name=f"{color_a}_{shape_a}",
            object_1_color=color_a,
            object_1_shape=shape_a,
            object_2_name=f"{color_b}_{shape_b}",
            object_2_color=color_b,
            object_2_shape=shape_b,
        )

        # Counterfactual attributes: Swap colors of object 1 and object 2
        attr_cf = SceneAttributes(
            object_1_name=f"{color_b}_{shape_a}",
            object_1_color=color_b,
            object_1_shape=shape_a,
            object_2_name=f"{color_a}_{shape_b}",
            object_2_color=color_a,
            object_2_shape=shape_b,
        )

        img_orig_path, bbox1_o, bbox2_o = self.generate_scene(attr_orig, rel, f"sample_{idx:05d}_orig.png")
        img_cf_path, bbox1_c, bbox2_c = self.generate_scene(attr_cf, rel, f"sample_{idx:05d}_cf.png")

        attr_orig.object_1_bbox = bbox1_o
        attr_orig.object_2_bbox = bbox2_o
        attr_cf.object_1_bbox = bbox1_c
        attr_cf.object_2_bbox = bbox2_c

        q_orig = f"Is the {color_a} {shape_a} {rel.replace('_', ' ')} the {color_b} {shape_b}?"
        q_cf = f"Is the {color_b} {shape_a} {rel.replace('_', ' ')} the {color_a} {shape_b}?"

        sample_orig = SpatialSample(
            sample_id=f"sample_{idx:05d}_orig",
            image_path=img_orig_path,
            question=q_orig,
            answer="Yes",
            spatial_relation=rel,
            attributes=attr_orig,
            metadata={"is_counterfactual": False, "pair_index": idx}
        )

        sample_cf = SpatialSample(
            sample_id=f"sample_{idx:05d}_cf",
            image_path=img_cf_path,
            question=q_cf,
            answer="Yes",
            spatial_relation=rel,
            attributes=attr_cf,
            metadata={"is_counterfactual": True, "pair_index": idx}
        )

        return CounterfactualPair(
            pair_id=f"pair_{idx:05d}",
            sample_original=sample_orig,
            sample_counterfactual=sample_cf,
            swapped_attribute="colors"
        )

    def generate_dataset(self, num_pairs: int = 100) -> List[CounterfactualPair]:
        """Generate a dataset of counterfactual pairs and write metadata to disk."""
        pairs = []
        metadata_file = self.output_dir / "metadata" / "synthetic_metadata.json"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)

        exported_records = []
        for i in range(num_pairs):
            pair = self.generate_counterfactual_pair(i)
            pairs.append(pair)
            exported_records.append(pair.to_dict())

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(exported_records, f, indent=2)

        return pairs
