from typing import Dict, Any, List

try:
    import numpy as np
except ImportError:
    np = None


class SpatialReasoningEvaluator:
    """Evaluates accuracy of spatial relation classification and VLM predictions."""

    def __init__(self):
        pass

    def evaluate_predictions(
        self,
        predictions: List[str],
        ground_truths: List[str],
        spatial_relations: List[str]
    ) -> Dict[str, Any]:
        """Compute overall accuracy and breakdown per spatial relation."""
        correct = [p.strip().lower() == gt.strip().lower() for p, gt in zip(predictions, ground_truths)]
        overall_acc = float(sum(correct) / max(1, len(correct)))

        relation_breakdown = {}
        unique_relations = sorted(list(set(spatial_relations)))
        for rel in unique_relations:
            rel_indices = [i for i, r in enumerate(spatial_relations) if r == rel]
            if rel_indices:
                rel_correct = [correct[i] for i in rel_indices]
                rel_acc = float(sum(rel_correct) / len(rel_correct))
                relation_breakdown[rel] = rel_acc

        return {
            "overall_accuracy": overall_acc,
            "relation_accuracy": relation_breakdown,
        }
