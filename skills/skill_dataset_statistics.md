# Skill 20: L1 vs. L2 Dataset Descriptive Statistics & EDA

## Context & Objective
Before presenting the STOCK-T algorithm results in our NeurIPS paper, we must provide an Exploratory Data Analysis (EDA) section. We need to statistically demonstrate the behavioral differences between Native (L1) and Bilingual (L2) readers in the GECO dataset. This justifies our design choices in the Psycholinguistic-Oculomotor Model (POM), specifically the regression penalties and saccade momentum.

## Required Implementation

### 1. Statistical Analysis Script (`scripts/geco/analyze_dataset_stats.py`)
Write a script that parses the original, clean GECO dataset (before adding any artificial drift). Select 5 L1 subjects and 5 L2 subjects across 10 reading trials.

Calculate the following 4 metrics for each group (L1 vs. L2):
1.  **Average Fixation Duration (ms)**
2.  **Skipping Rate (%)**: Proportion of words in a sentence that received zero fixations.
3.  **Regression Rate (%)**: Proportion of saccades that moved backward (to a smaller word index).
4.  **Average Forward Saccade Amplitude (in characters/words)**.

### 2. Output & Visualization
1. Print a markdown table comparing the Means and Standard Deviations for L1 vs. L2 on these 4 metrics.
2. Generate a grouped Bar Chart using `matplotlib` or `seaborn` comparing the 4 metrics between L1 and L2. Save it to `docs/figures/dataset_eda_stats.png`.
3. Save the markdown table and brief insights to `docs/experiments/2026-05-02_Dataset_Statistics.md`.