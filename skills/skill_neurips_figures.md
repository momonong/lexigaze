# Skill 24: NeurIPS Publication-Ready Visualizations

## Context & Objective
We need to generate high-quality, perfectly scaled vector graphics (.pdf) for a NeurIPS 2026 submission. The visual style must conform to single-column layout constraints (max width ~5.5 inches) and use 10pt Times New Roman fonts. 

## Instructions for the Agent
1. Read the base script provided at `generate_neurips_figures.py`.
2. Inspect the actual CSV/XLSX files located in `data/geco/`.
3. Fill in the `TODO` sections of the Python script with pandas/matplotlib logic to generate the exact charts described below.
4. Execute the script to output the `.pdf` files into `docs/NeurIPS/figures/`.

## Figure Requirements & Data Mapping

**Figure 1: Dataset EDA Stats (`dataset_eda_stats.pdf`)**
- **Data**: Infer summary stats from `data/geco/L1ReadingData.xlsx` and `data/geco/L2ReadingData.xlsx` (or related summary files).
- **Goal**: Grouped Bar Chart comparing Fixation Duration and Regression Rate for L1 vs L2.
- **Dimensions**: `figsize=(5.5, 2.0)` (Wide format).

**Figure 2: Noise Robustness Chart (`noise_robustness_chart.pdf`)**
- **Data**: `data/geco/noise_stress_results.csv`
- **Goal**: Line Chart. X-axis = Vertical Drift (px), Y-axis = Accuracy (%). Lines for Baseline vs STOCK-T.
- **Dimensions**: `figsize=(5.5, 2.0)` (Wide format). Highlight x=45px with a vertical dashed line.

**Figure 3: Qualitative Trajectory Correction (`trial5_analysis.pdf`)**
- **Data**: `data/geco/geco_l1_pp01_trial5_clean.csv`
- **Goal**: 2D Trajectory plot showing raw noisy gaze vs STOCK-T corrected gaze snapping to text boxes.
- **Dimensions**: `figsize=(5.5, 1.8)` (Extra wide/flat format to show a sentence line).

**Figure 4: OVP Proficiency Correlation (`ovp_proficiency_correlation.pdf`)**
- **Data**: `data/geco/geco_l1_final_evaluation.csv` and L2 equivalents.
- **Goal**: Scatter Plot with a regression line. X-axis = Fixation Duration (proxy for cognitive load), Y-axis = Delta Accuracy (Center - OVP). Color code L1 vs L2.
- **Dimensions**: `figsize=(3.5, 2.8)` (Squarish, to fit alongside discussion text).