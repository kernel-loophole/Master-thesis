from typing import Dict, Any, List

try:
    import numpy as np
except ImportError:
    np = None


class CausalImpactEvaluator:
    """Evaluates causal steering interventions on target spatial concepts and collateral semantic effects (RQ4)."""

    def __init__(self):
        pass

    def evaluate_intervention_causality(
        self,
        clean_outputs: List[str],
        steered_outputs: List[str],
        target_relations: List[str],
        expected_steered_relations: List[str],
        object_attributes: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculates:
        1. Intended effect rate (did steering change prediction to target relation?)
        2. Specificity score (did unrelated semantic attributes remain preserved?)
        3. Collateral semantic shift (frequency of unintended semantic mutations)
        4. Intervention efficiency
        """
        intended_success = []
        attribute_preservation = []

        for clean, steered, target_rel, exp_rel, attr in zip(
            clean_outputs, steered_outputs, target_relations, expected_steered_relations, object_attributes
        ):
            # 1. Intended Effect
            is_intended = exp_rel.lower() in steered.lower()
            intended_success.append(1.0 if is_intended else 0.0)

            # 2. Collateral Attribute Preservation (e.g. color / object identity preserved in text)
            color1 = attr.get("object_1_color", "")
            shape1 = attr.get("object_1_shape", "")

            color1_preserved = color1.lower() in steered.lower() if color1 else True
            shape1_preserved = shape1.lower() in steered.lower() if shape1 else True

            attribute_preservation.append(1.0 if (color1_preserved and shape1_preserved) else 0.0)

        intended_effect_rate = float(sum(intended_success) / max(1, len(intended_success)))
        specificity_score = float(sum(attribute_preservation) / max(1, len(attribute_preservation)))
        collateral_semantic_shift = 1.0 - specificity_score

        return {
            "intended_effect_rate": intended_effect_rate,
            "specificity_score": specificity_score,
            "collateral_semantic_shift": collateral_semantic_shift,
            "steering_efficiency_index": intended_effect_rate * specificity_score
        }
