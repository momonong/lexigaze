# Skill 18: Cross-Subject Benchmark & OVP Analysis (L1 vs. L2)

## Context & Objective
To prove the generalizability of the LexiGaze (STOCK-T) architecture, we must evaluate it across multiple subjects, not just `pp01`. Furthermore, we want to validate the "OVP Anomaly" discovered in the ablation study: the hypothesis that Native (L1) readers benefit from Optimal Viewing Position (OVP) offsets, while Second-Language (L2) readers perform better when fixating on the geometric center of words due to higher cognitive load.

## Required Implementation

### 1. Multi-Subject Evaluation Script (`scripts/geco/evaluate_population.py`)
Write a script that iterates over a selected pool of subjects from the GECO dataset:
- **L1 Group**: Select 5 native English speakers (e.g., `L1_01` to `L1_05`).
- **L2 Group**: Select 5 bilingual readers (e.g., `pp01` to `pp05`).
*(If the dataset structure allows, run across all available subjects).*

For each subject, evaluate 10 trials under the standard +45px vertical drift. 
Test two specific configurations:
- **Model 4 (Center)**: STOCK-T (POM + Multi-Hypothesis EM) with NO OVP offset.
- **Model 5 (OVP)**: Ultimate STOCK-T with the 35% OVP offset.

### 2. Output and Aggregation
Compute the **Average Strict Accuracy** for both models, grouped by L1 vs. L2.
Apply a simple Paired t-test (using `scipy.stats.ttest_rel`) to check if the difference between Model 4 and Model 5 is statistically significant within each group.

### 3. Output (@docs)
* Generate a Markdown report: `docs/experiments/2026-05-02_Cross_Subject_OVP_Analysis.md`.
* The report must include a table comparing the Mean Accuracy of (Model 4 vs Model 5) for the L1 Group and the L2 Group, along with the calculated p-values.