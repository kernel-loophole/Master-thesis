# Interpreting and Steering Spatial Reasoning via Multimodal Sparse Autoencoders (SAEs)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

A research codebase for investigating whether **Multimodal Sparse Autoencoders (SAEs)** can identify and causally steer disentangled spatial-reasoning features inside vision-language models (VLMs).

---

## 1. Project Overview & Research Motivation

Linear probing and direct linear interventions ($h_t' = h_t + \alpha \cdot \text{probe}$) in vision-language models can produce collateral semantic changes because learned linear directions are frequently entangled with non-target properties such as **color, texture, object identity, shape, background, and language attributes**.

This research repository investigates whether Sparse Autoencoders (SAEs) provide a more interpretable, disentangled feature basis for spatial concepts (such as *containment, inside/outside, left/right, above/below, front/behind, depth ordering*).

---

## 2. Core Research Questions

- **RQ1 — Representation:** Where inside a vision-language model are spatial relations represented? *(Vision encoder, projector, multimodal token embeddings, intermediate residual streams, attention/MLP outputs, final hidden states).*
- **RQ2 — Feature Discovery:** Can sparse autoencoders discover interpretable features corresponding to spatial concepts without hardcoding 1-to-1 mapping assumptions?
- **RQ3 — Disentanglement:** Are SAE features less entangled with irrelevant visual attributes than ordinary linear probe directions?
- **RQ4 — Causal Steering:** Can individual SAE features be causally manipulated ($z_i' = z_i + \alpha$ or $z_i' = c$) to change spatial reasoning predictions while minimizing collateral semantic changes?

---

## 3. Current Project Status

> [!IMPORTANT]
> **Status:** Experimental Research Framework.
> This repository is designed for hypothesis testing and controlled falsification. Empirical conclusions regarding SAE superiority or feature monosemanticity have not yet been established and are subject to ongoing experimental verification.

---

## 4. System Architecture

```
                    ┌─────────────────────────┐
                    │ Synthetic Scene Data    │
                    │ & Counterfactual Pairs  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Vision-Language Model   │
                    │      (VLM Adapter)      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Activation Extraction   │
                    │ & Layer Storage         │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Sparse Autoencoder      │
                    │      (SAE Training)     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Feature Discovery &     │
                    │ Correlation Analysis    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Causal Steering         │
                    │ Intervention Engine     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Disentanglement &       │
                    │ Causal Evaluation       │
                    └─────────────────────────┘
```

---

## 5. Directory Structure

```
.
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── .env.example
├── configs/
│   ├── model/
│   ├── data/
│   ├── sae/
│   ├── extraction/
│   ├── intervention/
│   └── experiments/
├── src/
│   └── spatial_sae/
│       ├── data/
│       ├── models/
│       ├── extraction/
│       ├── sae/
│       ├── probing/
│       ├── intervention/
│       ├── evaluation/
│       ├── experiments/
│       └── utils/
├── scripts/
│   ├── prepare_data.py
│   ├── extract_activations.py
│   ├── train_sae.py
│   ├── analyze_features.py
│   ├── run_probe.py
│   └── run_intervention.py
├── tests/
├── notebooks/
├── data/
├── checkpoints/
├── outputs/
└── docs/
```

---

## 6. Quick Start (Synthetic Dry Run)

You can run a complete end-to-end synthetic pipeline (dataset generation, dry-run extraction, SAE training, feature discovery, and causal intervention) without downloading a VLM model:

### Step 1: Generate Synthetic Dataset & Counterfactual Pairs

```bash
python scripts/prepare_data.py --config configs/data/default.yaml --num_pairs 100
```

### Step 2: Extract Activations (Dry Run with Mock VLM)

```bash
python scripts/extract_activations.py --config configs/extraction/default.yaml --dry_run
```

### Step 3: Train Sparse Autoencoder (SAE)

```bash
python scripts/train_sae.py --config configs/sae/default.yaml
```

### Step 4: Discover Features & Analyze Correlations

```bash
python scripts/analyze_features.py --config configs/sae/default.yaml
```

### Step 5: Run Linear Probe Baseline vs Causal Steering Interventions

```bash
# Linear Probe Baseline
python scripts/run_probe.py --config configs/experiments/default.yaml

# SAE Causal Steering & Compositional Binding Experiment
python scripts/run_intervention.py --config configs/intervention/default.yaml --compositional
```

---

## 7. Real VLM Model Experiments

To configure and run experiments on a real open-source Vision-Language Model (such as `Qwen/Qwen2-VL-7B-Instruct` or `llava-hf/llava-1.5-7b-hf`):

1. **Edit Model Config (`configs/model/default.yaml`):**
   ```yaml
   model_name: "Qwen/Qwen2-VL-7B-Instruct"
   adapter_type: "huggingface"
   device: "cuda"
   dtype: "bfloat16"
   target_layer: "model.layers.16"
   ```

2. **Extract Activations from Real VLM:**
   ```bash
   python scripts/extract_activations.py --config configs/extraction/default.yaml --model_config configs/model/default.yaml
   ```

3. **Train SAE & Run Interventions:**
   ```bash
   python scripts/train_sae.py --config configs/sae/default.yaml
   python scripts/run_intervention.py --config configs/intervention/default.yaml
   ```

---

## 8. Documentation

- [Research Plan & Hypotheses](docs/research_plan.md)
- [Repository Architecture](docs/architecture.md)
- [Experiment Guide](docs/experiments.md)
- [Feature Taxonomy & Labeling Rules](docs/feature_taxonomy.md)

---

## 9. License

Distributed under the [MIT License](LICENSE).
