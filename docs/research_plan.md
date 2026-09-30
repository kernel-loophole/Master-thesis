# Comprehensive Research Plan

## Interpreting and Steering Spatial Reasoning via Multimodal Sparse Autoencoders (SAEs)

### 1. Motivation
Linear probing and direct linear interventions ($h_t' = h_t + \alpha \cdot \text{probe}$) in vision-language models (VLMs) can produce collateral semantic changes because learned linear directions are frequently entangled with non-target visual and linguistic properties, such as object color, shape, texture, identity, and background context. This project investigates whether Sparse Autoencoders (SAEs) can extract a more interpretable, disentangled feature basis that enables precise causal steering of spatial reasoning while preserving unrelated semantic attributes.

---

### 2. Hypotheses
- **H1 (Representation):** Spatial reasoning features in VLMs are localized in intermediate multimodal residual stream layers and vision-language projector representations rather than early visual encoders or final text generation tokens alone.
- **H2 (Sparse Feature Discovery):** Multimodal SAEs can discover sparse, highly selective latent features corresponding to primitive spatial concepts (containment, adjacency, relative positioning, depth ordering).
- **H3 (Disentanglement):** Individual SAE features exhibit significantly lower attribute leakage and higher purity than direct linear probe direction vectors when tested on counterfactual scene pairs.
- **H4 (Causal Steering):** Intervening on sparse SAE features ($z_i' = z_i + \alpha$ or $z_i' = c$) steer spatial predictions with higher target specificity and lower collateral semantic mutation than linear probe steering.

---

### 3. Core Research Questions

#### RQ1 — Representation
Where inside a vision-language model are spatial relations represented?
- Target activation sources: Vision encoder outputs, vision-language projector, multimodal token embeddings, intermediate transformer residual streams, selected attention/MLP layer outputs.

#### RQ2 — Feature Discovery
Can sparse autoencoders discover interpretable features corresponding to spatial concepts (inside, outside, left/right, above/below, front/behind, near/far, containment, overlap)?
- Experimental design allows 1-to-many, many-to-1, distributed, and polysemantic feature mappings without hardcoding 1-to-1 assumptions.

#### RQ3 — Disentanglement
Are SAE features less entangled with irrelevant attributes than ordinary linear probe directions?
- Evaluates confounds: object color, shape, texture, object identity, background, language templates, image position, object size.

#### RQ4 — Causal Steering
Can individual SAE features be causally manipulated to alter spatial reasoning while minimizing collateral semantic shift?
- Compares intervention strategies (`add`, `clamp`, `suppress`, `scale`) in SAE latent space ($h \to z \to z' \to h'$).

---

### 4. Experimental Variables
- **Independent Variables:**
  - Intervention method: Linear probe vector vs. SAE latent feature vs. Random feature vs. Matched control feature.
  - Intervention strength ($\alpha \in [-10.0, 10.0]$).
  - Target layer / representation source.
  - Scene attribute variations (colors, shapes, background grid, counterfactual attribute swaps).

- **Dependent Variables:**
  - Target spatial relation prediction accuracy.
  - Feature selectivity & purity index.
  - Counterfactual invariance score.
  - Specificity score & collateral semantic shift.
  - KL divergence of output logit distributions.

---

### 5. Baselines
1. **Unintervened Model Baseline:** Clean forward pass.
2. **Linear Probe Intervention Baseline:** Direct linear direction steering ($h' = h + \alpha \cdot w_{\text{probe}}$).
3. **Random Feature Steering Control:** Modifying arbitrary random SAE features with equal norm.
4. **Matched-Control Feature Steering:** Modifying correlated non-spatial SAE features (e.g., color-selective features).

---

### 6. Metrics
- **Spatial Accuracy:** Correctness of target spatial relation classification.
- **Spatial Selectivity & Purity:** Correlation with spatial label vs. max correlation with non-spatial attribute confounds.
- **Counterfactual Invariance:** Cosine similarity of feature activations between counterfactual scene pairs ($A: \text{red ball in blue box}$ vs $B: \text{blue ball in red box}$).
- **Intended Effect Rate:** Percentage of steered forward passes where model prediction matches targeted spatial relation.
- **Specificity & Collateral Shift:** Frequency of unintended mutations to object identity/color bindings.

---

### 7. Ablations
- Varying SAE expansion factor ($4\times, 8\times, 16\times$).
- Varying SAE sparsity loss formulation ($L_1$ penalty vs. Top-K selection).
- Steering at single layer vs. multi-layer simultaneous interventions.

---

### 8. Expected Failure Modes
- **Polysemantic Features:** Single SAE features responding to both spatial relation AND object identity.
- **Distributed Spatial Concepts:** Spatial concept represented across a linear combination of multiple SAE features rather than a single monosemantic unit.
- **Reconstruction Degradation:** High SAE reconstruction error distorting general language generation capabilities.

---

### 9. Reproducibility
- Deterministic seeding (`set_seed()`) across Python, NumPy, and PyTorch.
- Explicit configuration tracking (`configs/**/*.yaml` saved alongside all experiment outputs).
- Full metadata recorded for extracted activations, trained checkpoints, and evaluation runs.
