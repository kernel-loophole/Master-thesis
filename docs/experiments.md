# Experiment Protocols & Reproduction Guide

This guide details how to reproduce all experiments in the repository.

---

## 1. Prepare Synthetic Dataset & Counterfactual Pairs

Generate synthetic 2D scenes and matched counterfactual pairs:

```bash
python scripts/prepare_data.py --config configs/data/default.yaml --num_pairs 200
```

Exports images to `data/raw/synthetic_images/` and metadata to `data/metadata/synthetic_metadata.json`.

---

## 2. Extract Layer Activations

Extract hidden activations across target VLM layers:

```bash
# Dry run with Mock VLM Adapter
python scripts/extract_activations.py --config configs/extraction/default.yaml --dry_run

# Real VLM extraction
python scripts/extract_activations.py --config configs/extraction/default.yaml --model_config configs/model/default.yaml
```

Saves activation shards and metadata to `outputs/activations/`.

---

## 3. Train Sparse Autoencoder (SAE)

Train SAE on extracted activations:

```bash
python scripts/train_sae.py --config configs/sae/default.yaml
```

Saves checkpoints to `checkpoints/` and metrics to `outputs/sae/`.

---

## 4. Run Feature Discovery & Correlation Analysis

Discover top spatial concept features and evaluate purity:

```bash
python scripts/analyze_features.py --config configs/sae/default.yaml
```

Outputs feature correlations and purity scores to `outputs/sae/features/discovered_features.json`.

---

## 5. Train Linear Probe Baseline

Train direct linear probe baseline:

```bash
python scripts/run_probe.py --config configs/experiments/default.yaml
```

Saves probing accuracy and attribute leakage to `outputs/probes/baseline_results.json`.

---

## 6. Run Causal Steering Interventions

Execute causal steering in SAE latent space:

```bash
# Standard steering intervention
python scripts/run_intervention.py --config configs/intervention/default.yaml

# Compositional binding experiment (5 conditions)
python scripts/run_intervention.py --config configs/intervention/default.yaml --compositional
```

Outputs causal metrics to `outputs/interventions/` and `outputs/experiments/compositional_binding/`.
