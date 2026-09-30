from typing import Dict, Any, List, Tuple, Optional

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None


def get_top_activating_examples(
    feature_id: int,
    sae: Any,
    activations: Any,
    metadata_list: List[Dict[str, Any]],
    top_k: int = 10
) -> List[Tuple[float, Dict[str, Any]]]:
    """Retrieve top-K activating dataset examples and activation magnitudes for a specific SAE feature."""
    if torch is None or not isinstance(activations, torch.Tensor) or np is None:
        return []

    sae.eval()
    with torch.no_grad():
        z = sae.encode(activations.to(next(sae.parameters()).device))
        feature_acts = z[:, feature_id].cpu().numpy()

    top_indices = np.argsort(feature_acts)[::-1][:top_k]
    results = []
    for idx in top_indices:
        act_val = float(feature_acts[idx])
        meta = metadata_list[idx] if idx < len(metadata_list) else {"sample_id": f"idx_{idx}"}
        results.append((act_val, meta))

    return results


def feature_label_correlations(
    z_matrix: Any,
    labels: List[str]
) -> Dict[int, Dict[str, float]]:
    """Compute point-biserial / Pearson correlation between each SAE feature and discrete spatial concept labels."""
    if np is None or not isinstance(z_matrix, np.ndarray):
        return {}

    unique_labels = sorted(list(set(labels)))
    feature_correlations = {}

    num_features = z_matrix.shape[1]
    for feat_id in range(num_features):
        feat_acts = z_matrix[:, feat_id]
        corr_dict = {}
        for label in unique_labels:
            binary_target = np.array([1.0 if l == label else 0.0 for l in labels])
            if np.std(feat_acts) > 1e-8 and np.std(binary_target) > 1e-8:
                corr = np.corrcoef(feat_acts, binary_target)[0, 1]
                corr_dict[label] = float(np.nan_to_num(corr))
            else:
                corr_dict[label] = 0.0
        feature_correlations[feat_id] = corr_dict

    return feature_correlations


def feature_attribute_correlations(
    z_matrix: Any,
    attributes_list: List[Dict[str, Any]],
    target_attribute_key: str = "object_1_color"
) -> Dict[int, Dict[str, float]]:
    """Compute correlation between SAE features and non-spatial confounding semantic attributes (color, shape, etc.)."""
    attribute_values = [attr.get(target_attribute_key, "") for attr in attributes_list]
    return feature_label_correlations(z_matrix, attribute_values)


def rank_features_for_concept(
    z_matrix: Any,
    labels: List[str],
    target_concept: str,
    top_n: int = 10
) -> List[Tuple[int, float]]:
    """Rank SAE feature indices by selectivity score for a target spatial concept."""
    correlations = feature_label_correlations(z_matrix, labels)
    ranked = []
    for feat_id, corr_map in correlations.items():
        score = corr_map.get(target_concept, 0.0)
        ranked.append((feat_id, score))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def calculate_feature_purity(
    target_concept_corr: float,
    confound_corrs: List[float]
) -> float:
    """Calculate feature purity score: ratio of target spatial correlation to max confound attribute correlation."""
    max_confound = max(confound_corrs) if confound_corrs else 0.0
    if max_confound <= 0.0:
        return target_concept_corr
    return target_concept_corr / (max_confound + 1e-6)
