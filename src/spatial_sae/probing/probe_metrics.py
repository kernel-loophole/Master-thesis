from typing import Dict, Any, List

try:
    import numpy as np
except ImportError:
    np = None


def evaluate_probe_performance(
    y_true: List[int],
    y_pred: List[int],
    attribute_labels: Dict[str, List[Any]]
) -> Dict[str, float]:
    """Compute spatial prediction accuracy and attribute leakage for linear probe baselines."""
    if np is None:
        correct = [t == p for t, p in zip(y_true, y_pred)]
        accuracy = sum(correct) / max(1, len(correct))
        return {"spatial_accuracy": float(accuracy)}

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    accuracy = float(np.mean(y_true_arr == y_pred_arr))

    # Calculate attribute leakage: correlation between probe predictions and confound attribute labels (color/shape)
    attribute_leakage = {}
    for attr_name, attr_vals in attribute_labels.items():
        if len(attr_vals) == len(y_pred):
            unique_attrs = sorted(list(set(attr_vals)))
            corr_scores = []
            for u_attr in unique_attrs:
                binary_attr = np.array([1.0 if v == u_attr else 0.0 for v in attr_vals])
                if np.std(binary_attr) > 1e-8 and np.std(y_pred_arr) > 1e-8:
                    c = np.corrcoef(y_pred_arr, binary_attr)[0, 1]
                    corr_scores.append(abs(float(np.nan_to_num(c))))
            attribute_leakage[f"leakage_{attr_name}"] = float(np.max(corr_scores)) if corr_scores else 0.0

    metrics = {"spatial_accuracy": accuracy}
    metrics.update(attribute_leakage)
    return metrics
