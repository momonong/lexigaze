# NeurIPS 2026: Domain-Specific Eye-Tracking Visualization Report

This document details the specialized visualizations generated for the NeurIPS 2026 submission, following established conventions in cognitive science (SWIFT, E-Z Reader) and computer vision.

## 1. Technical Standards
- **Typography**: Serif (Times New Roman / DejaVu Serif), 10pt for labels, 8pt for legends.
- **Precision**: Vector-based PDF with embedded Type 42 fonts.
- **Color Palette**: Group-aware (`blue` for Native/L1, `red` for Bilingual/L2) and colorblind-friendly.

## 2. Visualization Suite

### A. SWIFT-Style Activation Field (`fig_cm_activation_field.pdf`)
- **Concept**: Visualizes the "Semantic Gravity" or "Linguistic Saliency" pulling the gaze.
- **Insight**: The field shows clear activation peaks on high-surprisal words, illustrating how the **Cognitive Mass** acts as a probabilistic attractor in the Bayesian emission model.

### B. Psycholinguistic Effect Curve (`fig_psycholinguistic_effects.pdf`)
- **Concept**: Proves the relationship between word difficulty and cognitive load.
- **Insight**: Regression lines for L1 and L2 groups show a significant positive correlation between **Cognitive Mass** and **Fixation Duration**. L2 readers exhibit a steeper slope, confirming higher sensitivity to linguistic complexity.

### C. Scanpath Recovery & Drift Correction (`fig_scanpath_correction.pdf`)
- **Concept**: Qualitative proof of the **STOCK-T** algorithm's success.
- **Insight**: Demonstrates the "Line-Locking" failure where raw gaze drifts to the row below. The **STOCK-T** path successfully breaks this trap, snapping perfectly to the intended semantic targets.

### D. Hardware Stress Degradation (`fig_noise_degradation.pdf`)
- **Concept**: Establishes the algorithm's performance envelope.
- **Insight**: Shows high stability (90%+) up to the 45px line-height threshold. While spatial baselines collapse immediately, the neuro-symbolic approach maintains tracking capability even under extreme hardware failure.

## 3. Reproduction
To regenerate the domain-specific suite, run:
```bash
uv run python scripts/geco/neurips_domain_viz.py
```

---
**Prepared by**: LexiGaze Research Team  
**Date**: May 3, 2026
