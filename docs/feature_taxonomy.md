# Spatial Relation Taxonomy & Feature Labeling Guidelines

This document outlines the taxonomy of spatial concepts investigated in this research repository and guidelines for feature labeling.

---

## 1. Spatial Relation Taxonomy

### A. Primitive Topological Relations
- **Containment / Inside:** Object A enclosed within Object B (`inside`, `contained_in`).
- **Separation / Outside:** Object A located outside Object B (`outside`, `disjoint`).
- **Overlap:** Object A boundary partially intersects Object B boundary (`overlapping`, `intersecting`).
- **Adjacency / Touch:** Object A physically touches Object B without overlap (`adjacent_to`, `touching`).

### B. Directional & Relative Positioning
- **Left / Right:** Horizontal relative ordering (`left_of`, `right_of`).
- **Above / Below:** Vertical relative ordering (`above`, `below`).
- **Front / Behind:** Depth / occlusion ordering (`front_of`, `behind`).
- **Near / Far:** Metric proximity (`near`, `far`).

### C. 3D & Elevation Attributes
- **Elevation / Altitude:** Higher vs lower relative surface plane (`elevated`, `grounded`).
- **Depth Layering:** Foregound, midground, background ordering (`depth_order`).

---

## 2. Feature Labeling Rules & Scientific Terminology

When analyzing SAE latent features, adhere strictly to cautious scientific terminology:

- Do **NOT** label an unverified feature as "monosemantic".
- Use precise terms:
  - **Candidate Spatial Feature:** A latent SAE unit demonstrating statistically significant correlation ($r > 0.4$) with a target spatial concept.
  - **Feature Selectivity Score:** Ratio of target spatial correlation to mean background correlation.
  - **Feature Purity Score:** Ratio of target spatial correlation to max confounding attribute correlation (e.g. object color or shape).
  - **Polysemantic Feature:** A feature unit exhibiting high correlation with multiple distinct semantic attributes simultaneously.
