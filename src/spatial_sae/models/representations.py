from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple


class RepresentationSource(str, Enum):
    """Sources of representations inside a Vision-Language Model (RQ1)."""
    VISION_ENCODER = "vision_encoder"
    PROJECTOR = "projector"
    MULTIMODAL_TOKENS = "multimodal_tokens"
    RESIDUAL_STREAM = "residual_stream"
    ATTENTION_OUTPUT = "attention_output"
    MLP_OUTPUT = "mlp_output"
    FINAL_HIDDEN = "final_hidden"


@dataclass
class ActivationLocation:
    """Defines exact layer and position specification for activation extraction."""
    source: RepresentationSource
    layer_name: str
    layer_index: int
    token_slice: Optional[slice] = None
    description: str = ""
