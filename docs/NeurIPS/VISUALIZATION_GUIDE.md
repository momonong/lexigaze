# LexiGaze: NeurIPS Visualization Guide (v2)

This document provides technical details for the publication-ready visualizations generated for the LexiGaze research paper. All figures are generated using `scripts/geco/generate_neurips_figures_v2.py` and strictly adhere to NeurIPS 2026 formatting standards.

## 1. Technical Standards
- **Typography**: Serif (Times New Roman / DejaVu Serif), 10pt for labels/titles, 8pt for ticks/legends.
- **Dimensions**: Full text width (5.5 inches) for horizontal visibility.
- **Format**: Vector-based (PDF) with embedded Type 42 fonts for high-quality LaTeX printing.
- **Color Palette**: Colorblind-friendly `sns.colorblind`.

## 2. Figure Descriptions

### Figure 1: Performance Degradation under Systematic Hardware Drift (`fig1_noise_robustness.pdf`)
- **Objective**: Demonstrate the robustness of the STOCK-T algorithm across a wide range of hardware failures.
- **Key Insight**: While baseline methods collapse at 30px drift, STOCK-T maintains stable 90%+ accuracy up to 45px (the GECO standard noise floor) and gracefully degrades at extreme offsets.

### Figure 2: OVP Anomaly & Reading Proficiency (`fig2_ovp_anomaly.pdf`)
- **Objective**: Multi-panel analysis showing the correlation between cognitive load and gaze targeting preference.
- **Panel (a)**: Linear regression showing that longer fixation durations (lower proficiency) correlate with a stronger reliance on geometric centers.
- **Panel (b)**: Boxplot illustrating the distinct "Center-Gravity" preference in Bilingual (L2) vs. Native (L1) groups.

### Figure 3: Ablation Study: Sequential Contribution of Modular Innovations (`fig3_ablation_study.pdf`)
- **Objective**: Prove the necessity of the Psycholinguistic-Oculomotor Model (POM) as the "Enabler" for hardware calibration.
- **Key Insight**: Rule-based Viterbi fails to initialize calibration correctly under extreme drift. Only the inclusion of POM (STOCK-T) provides the "Linguistic Confidence" needed to overcome the Line-Locking failure mode, resulting in a ~45% absolute accuracy leap.

### Figure 4: Gaze Kinematics & Transition Matrix Heatmap (`fig4_kinematics_pom.pdf`)
- **Objective**: Illustrate the underlying biological and linguistic priors.
- **Panel (a)**: Relative ratio of gaze metrics (L2 vs L1) justifying the adaptive transition parameters.
- **Panel (b)**: A heatmap visualization of the POM transition matrix, showing the forward-biased momentum and sparse regression anchors.

## 3. How to Regenerate
To regenerate these figures, ensure you are in the project root and run:
```bash
uv run python scripts/geco/generate_neurips_figures_v2.py
```

---
**Prepared by**: LexiGaze Research Team  
**Date**: May 3, 2026
