from typing import Dict, Any, List, Tuple

try:
    import numpy as np
except ImportError:
    np = None


class DisentanglementEvaluator:
    """Evaluates disentanglement of SAE features vs linear probe directions across confounding attributes (RQ3)."""

    def __init__(self):
        pass

    def compute_selectivity_scores(
        self,
        feature_activations: Any,
        spatial_labels: List[str],
        color_labels: List[str],
        shape_labels: List[str],
        object_labels: List[str]
    ) -> Dict[str, Any]:
        """
        Calculates feature selectivity for spatial relation vs confounds:
        - spatial_selectivity
        - color_selectivity
        - shape_selectivity
        - object_selectivity
        """
        if np is None or not isinstance(feature_activations, np.ndarray):
            return {}

        num_features = feature_activations.shape[1]
        results = {}

        for feat_id in range(num_features):
            acts = feature_activations[:, feat_id]
            if np.std(acts) < 1e-8:
                continue

            sp_corr = self._max_label_correlation(acts, spatial_labels)
            col_corr = self._max_label_correlation(acts, color_labels)
            shp_corr = self._max_label_correlation(acts, shape_labels)
            obj_corr = self._max_label_correlation(acts, object_labels)

            disentanglement_score = sp_corr - max(col_corr, shp_corr, obj_corr)

            results[feat_id] = {
                "spatial_selectivity": sp_corr,
                "color_selectivity": col_corr,
                "shape_selectivity": shp_corr,
                "object_selectivity": obj_corr,
                "disentanglement_score": disentanglement_score,
            }

        return results

    def compute_counterfactual_invariance(
        self,
        acts_orig: Any,
        acts_cf: Any
    ) -> float:
        """
        Compute feature stability under counterfactual attribute swaps (e.g. red ball in blue box vs blue ball in red box).
        Higher score (closer to 1.0) means feature tracks spatial relation independently of swapped attributes.
        """
        if np is None or not isinstance(acts_orig, np.ndarray) or not isinstance(acts_cf, np.ndarray):
            return 0.0

        if acts_orig.shape != acts_cf.shape or acts_orig.size == 0:
            return 0.0

        dot_prod = np.sum(acts_orig * acts_cf, axis=-1)
        norm_orig = np.linalg.norm(acts_orig, axis=-1)
        norm_cf = np.linalg.norm(acts_cf, axis=-1)

        sim = dot_prod / (norm_orig * norm_cf + 1e-8)
        return float(np.mean(sim))

    def _max_label_correlation(self, acts: Any, labels: List[str]) -> float:
        if np is None or not labels:
            return 0.0
        unique_labels = sorted(list(set(labels)))
        corrs = []
        for ul in unique_labels:
            binary = np.array([1.0 if l == ul else 0.0 for l in labels])
            if np.std(binary) > 1e-8:
                c = np.corrcoef(acts, binary)[0, 1]
                corrs.append(abs(float(np.nan_to_num(c))))
        return float(max(corrs)) if corrs else 0.0
