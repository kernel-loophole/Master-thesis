import json
from pathlib import Path
from typing import Dict, Any

from spatial_sae.utils.config import ConfigDict
from spatial_sae.utils.reproducibility import set_seed
from spatial_sae.utils.logging import get_logger
from spatial_sae.models.vlm import MockVLMAdapter
from spatial_sae.sae.model import SparseAutoencoder
from spatial_sae.intervention.sae_intervention import SAEInterventionEngine
from spatial_sae.evaluation.causal_metrics import CausalImpactEvaluator

logger = get_logger("spatial_sae.exp_intervene")


def run_causal_interventions(config: ConfigDict) -> Dict[str, Any]:
    """Execute causal steering intervention experiment."""
    set_seed(config.get("seed", 42))
    output_dir = Path(config.get("output_dir", "./outputs/interventions"))
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dim = config.get("input_dim", 512)
    latent_dim = config.get("latent_dim", 2048)

    vlm = MockVLMAdapter(hidden_dim=input_dim)
    vlm.load_model("mock")

    sae = SparseAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    engine = SAEInterventionEngine(vlm_adapter=vlm, sae=sae, target_layer="model.layers.16")

    feature_indices = config.get("feature_indices", [5, 12])
    mode = config.get("intervention_mode", "add")
    strength = config.get("intervention_strength", 5.0)

    logger.info(f"Running causal steering intervention: mode={mode}, features={feature_indices}, strength={strength}")

    strategy = engine.create_strategy(mode=mode, feature_indices=feature_indices, strength=strength)
    engine.register_intervention(strategy)

    # Run forward pass under intervention
    inputs = vlm.prepare_inputs([], ["Is the red ball inside the blue box?"])
    _ = vlm.forward(inputs)

    engine.remove_intervention()

    # Evaluate causal impact
    evaluator = CausalImpactEvaluator()
    results = evaluator.evaluate_intervention_causality(
        clean_outputs=["No"],
        steered_outputs=["Yes, the red ball is inside the blue box."],
        target_relations=["inside"],
        expected_steered_relations=["inside"],
        object_attributes=[{"object_1_color": "red", "object_1_shape": "ball"}]
    )

    out_file = output_dir / "causal_steering_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Causal Steering Intended Effect Rate: {results['intended_effect_rate']:.4f}")
    return results
