from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class SceneAttributes:
    """Attributes describing objects in a spatial scene."""
    object_1_name: str
    object_1_color: str
    object_1_shape: str
    object_2_name: str
    object_2_color: str
    object_2_shape: str
    object_1_bbox: Optional[Tuple[int, int, int, int]] = None
    object_2_bbox: Optional[Tuple[int, int, int, int]] = None
    background_color: str = "white"
    texture: Optional[str] = None


@dataclass
class SpatialSample:
    """Single multimodal spatial reasoning sample."""
    sample_id: str
    image_path: Optional[str]
    question: str
    answer: str
    spatial_relation: str  # e.g., 'inside', 'left_of', 'above', 'behind', etc.
    attributes: SceneAttributes
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class CounterfactualPair:
    """Pair of samples with matched spatial arrangement but swapped semantic attributes."""
    pair_id: str
    sample_original: SpatialSample
    sample_counterfactual: SpatialSample
    swapped_attribute: str  # e.g. 'colors', 'shapes', 'identities'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "swapped_attribute": self.swapped_attribute,
            "original": self.sample_original.to_dict(),
            "counterfactual": self.sample_counterfactual.to_dict()
        }


@dataclass
class ExtractionMetadata:
    """Metadata recorded during activation extraction for an individual sample/token."""
    sample_id: str
    image_id: Optional[str]
    question: str
    spatial_relation: str
    object_1_color: str
    object_1_shape: str
    object_2_color: str
    object_2_shape: str
    layer: str
    token_position: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
