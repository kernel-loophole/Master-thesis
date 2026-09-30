# Repository Architecture

## Pipeline Architecture

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

## Core Modules & Separation of Concerns

1. **`spatial_sae.data`**: Manages synthetic scene rendering, counterfactual pair generation, schemas, and PyTorch datasets.
2. **`spatial_sae.models`**: VLM adapters (`VLMAdapter`, `HuggingFaceVLMAdapter`, `MockVLMAdapter`), representation definitions, and forward hook managers.
3. **`spatial_sae.extraction`**: Layer activation collection, disk storage shards (`ActivationStorage`), and extraction pipelines (`ActivationExtractor`).
4. **`spatial_sae.sae`**: PyTorch Sparse Autoencoder (`SparseAutoencoder`), $L_1$ / Top-K loss functions (`SAELoss`), trainer (`SAETrainer`), metrics, and feature discovery tools.
5. **`spatial_sae.probing`**: Linear probe baselines (`LinearProbe`) and direction steering comparison ($h' = h + \alpha \cdot w_{\text{probe}}$).
6. **`spatial_sae.intervention`**: Steering intervention engine (`SAEInterventionEngine`), PyTorch forward steering hooks, and strategies (`add`, `clamp`, `suppress`, `scale`).
7. **`spatial_sae.evaluation`**: Quantitative metrics for spatial accuracy, feature selectivity, counterfactual invariance, specificity score, and KL divergence.
8. **`spatial_sae.experiments`**: Standardized experiment execution scripts and compositional binding evaluation.

---

## Extension Guide: Adding a New VLM

To add support for a new Vision-Language Model (e.g. LLaVA, PaliGemma, IDEFICS):
1. Subclass `spatial_sae.models.vlm.VLMAdapter`.
2. Implement `load_model()`, `prepare_inputs()`, `forward()`, `get_activation()`, `register_hook()`, and `remove_hooks()`.
3. Update `configs/model/default.yaml` with the target model name and layer names.
