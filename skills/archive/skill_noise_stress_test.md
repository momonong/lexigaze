# Skill 19: Noise Robustness Stress Test & Visualization

## Context & Objective
NeurIPS reviewers expect to see the breakdown point of the proposed algorithm. We will simulate varying degrees of hardware drift and plot the accuracy degradation curve to demonstrate the robustness of our Multi-Hypothesis EM and POM integration compared to baselines.

## Required Implementation

### 1. Stress Test Script (`scripts/geco/evaluate_noise_stress.py`)
Create a script that evaluates Subject `pp01` (over 10 trials) across an array of vertical drift levels:
`drift_levels = [0, 15, 30, 45, 60, 75]` (in pixels).

For each drift level, evaluate:
- **Baseline**: Config A (Nearest Bounding Box / Raw Proximity).
- **EM Only**: Model 2 (Viterbi + Multi-Hypothesis EM, No POM).
- **Ours (STOCK-T)**: Model 4 (POM + Multi-Hypothesis EM).

Record the **Average Strict Accuracy** for each model at each drift level.

### 2. Visualization (`scripts/geco/plot_stress_test.py`)
Write a Python script using `matplotlib` to generate a high-quality line chart:
- **X-axis**: Vertical Drift (Pixels) [0 to 75].
- **Y-axis**: Strict Accuracy (%) [0 to 100].
- **Lines**: 
  - Baseline (Dotted red line, rapidly decaying).
  - EM Only (Dashed orange line, failing at higher drifts due to line-locking).
  - STOCK-T (Solid bold blue line with circular markers, remaining highly stable).
- **Styling**: Academic style, clean grid, clear legend, and axis labels.

### 3. Output (@docs)
* Save the raw data table to `docs/experiments/2026-05-02_Noise_Stress_Test.md`.
* Save the plot to `docs/figures/noise_robustness_chart.png` or `.pdf`.