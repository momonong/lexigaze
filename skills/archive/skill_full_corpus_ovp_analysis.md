# Skill 21: Full-Corpus OVP & Proficiency Analysis

## Context & Objective
Our advisor highlighted that the \"OVP Anomaly\" (L2 readers benefiting from geometric center targeting rather than the biological Optimal Viewing Position) is a highly compelling finding. To solidify this for our NeurIPS submission, we need to scale our evaluation from a few subjects to the **entire available GECO dataset** (all L1 native speakers and all L2 bilingual learners).
We want to establish a robust statistical correlation between L1/L2 grouping (and ideally reading proficiency/speed) and their performance under OVP vs. Center-snapping configurations.

## Required Implementation

### 1. Full-Corpus Batch Script (`scripts/geco/evaluate_full_corpus.py`)
Write a robust Python script that iterates over all available L1 and L2 subjects in our processed GECO dataset.
For each subject, evaluate a representative sample of trials (e.g., 10-20 sentences) under the standard +45px vertical drift.

For each trial, compute the Strict Accuracy for:
1. **Model 4 (Center)**: STOCK-T (POM + Multi-Hypothesis EM) with NO OVP offset.
2. **Model 5 (OVP)**: STOCK-T with the 35% OVP offset.

### 2. Proficiency Metric Extraction
If the GECO dataset contains proficiency metrics (e.g., English comprehension scores, vocabulary size, or simply \"average reading speed\"), extract this metric for each subject. If explicit scores are unavailable, use the subject's **Global Average Fixation Duration** as an inverse proxy for proficiency (longer duration = higher cognitive load / lower proficiency).

### 3. Data Aggregation & Analysis
Calculate the $\\Delta$ Accuracy for each subject:
`Delta_Acc = Accuracy(Center) - Accuracy(OVP)`
*A positive Delta means the subject relies more on the geometric center (typical L2 behavior). A negative Delta means the subject benefits from OVP (typical L1 behavior).*

### 4. Output & Visualization (@docs)
1. **CSV Output**: Save the results to `docs/experiments/full_corpus_ovp_results.csv` (Columns: Subject, Group, Proficiency_Proxy, Center_Acc, OVP_Acc, Delta_Acc).
2. **Scatter Plot**: Use `matplotlib` to plot `Proficiency_Proxy` (X-axis) vs `Delta_Acc` (Y-axis). Color the dots by Group (L1 = Blue, L2 = Red). Add a trendline. Save to `docs/figures/ovp_proficiency_correlation.png`.
3. **Report**: Generate `docs/experiments/2026-05-02_Full_Corpus_OVP_Report.md` summarizing the statistical differences between the L1 and L2 cohorts.