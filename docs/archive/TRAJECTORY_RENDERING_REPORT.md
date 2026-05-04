# LexiGaze: Qualitative Trajectory Rendering Report

This report documents the implementation and outcome of **Skill 25: NeurIPS Qualitative Gaze Trajectory Renderer**. The objective was to provide a visual proof-of-concept for the paper, specifically illustrating how the **STOCK-T** architecture overcomes the "Line-Locking" failure mode caused by extreme hardware drift.

## 1. Visualizing the "Line-Locking" Failure
In edge-device eye-tracking, a systematic vertical drift of +45px (simulated in our study) often causes the gaze to fall closer to the text line *below* the actual target. 
- **Raw Gaze (Drift)**: As shown in `trial5_analysis.pdf`, the raw webcam data (red dashed line) is physically closer to the second line of text.
- **The Challenge**: A standard nearest-neighbor or rule-based sequence model would incorrectly "lock" onto the second line, reinforcing the hardware error.

## 2. The STOCK-T Solution
Our **STOCK-T** model uses a **Multi-Hypothesis EM Initialization (Skill 14)** combined with the **Psycholinguistic-Oculomotor Model (POM)** to break this trap.
- **Mechanism**: The system evaluates multiple line-start hypotheses and selects the one with the highest global Viterbi likelihood (determined by linguistic sequence flow).
- **Corrected Gaze (STOCK-T)**: The green solid line in the figure shows the trajectory after self-calibration. It successfully "snaps" back to the correct semantic path, accurately tracking the reader's intent despite the initial 45px error.

## 3. Figure Specifications
- **Dimensions**: 5.5" x 2.0" (Single column width).
- **Typography**: NeurIPS Serif (Times New Roman).
- **Markers**: Circular (Raw Drift) vs. Star (STOCK-T Corrected).
- **Aesthetics**: Minimalist, axis-free design focused on geometric relationships.

## 4. Path to Output
The final figures are saved at:
- `docs/NeurIPS/figures/fig1_kinematics.pdf` (Comparative EDA)
- `docs/NeurIPS/figures/fig2_ovp_anomaly.pdf` (Corpus-Scale Correlation)
- `docs/NeurIPS/figures/fig3_robustness.pdf` (Noise Stress Test)
- `docs/NeurIPS/figures/fig4_scanpath_recovery.pdf` (High-Fidelity Real Data)

## 5. Final Data-Driven Qualitative Scanpath
We have replaced all mock-data visualizations with authentic results derived from the **GECO L2 Corpus (Subject pp01, Trial 5)**.
- **Authentic Stimulus**: The text and word positions are extracted directly from the eye-tracking database, ensuring biological and linguistic realism.
- **Dynamic Calibration**: The figure demonstrates the **Multi-Hypothesis EM** correctly identifying a -40px vertical shift to resolve the "Line-Locking" failure mode induced by simulated hardware drift.
- **Visual Features**: Soft semantic gravity highlights, curved correction arcs, and directional saccade indicators provide a clear, professional narrative of the STOCK-T system's internal reasoning.

---
**Prepared by**: LexiGaze Research Orchestrator  
**Date**: May 3, 2026
