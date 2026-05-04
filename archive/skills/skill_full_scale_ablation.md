# Skill 16: Full-Scale Ablation Benchmark (Multiple Trials)

## Context & Objective
In our single-trial ablation study (Skill 15), Model 4 (without OVP) slightly outperformed Model 5 (with OVP) in Strict Accuracy (94.23% vs 92.31%). To determine if this is a statistical anomaly due to single-trial variance or a genuine architectural property, we must scale up our evaluation.
We will write a script to evaluate the same 5 ablation models across **multiple trials** (e.g., Trial 1 to Trial 10) for our L2 subject (`pp01`), and compute the **average Strict and Relaxed Accuracies**.

## Required Implementation

### 1. Multi-Trial Evaluation Script (`scripts/geco/evaluate_full_ablation.py`)
Create a new script that iterates through a list of trial IDs (e.g., `trials = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`) for the subject `pp01`.
For each trial, it should:
1. Apply the standard +45px vertical drift and Gaussian jitter.
2. Evaluate the 5 configurations defined in Skill 15:
   * **Model 1**: Base Viterbi
   * **Model 2**: Viterbi + Multi-Hypothesis EM
   * **Model 3**: Viterbi + EM + OVP
   * **Model 4**: STOCK-T (POM + EM)
   * **Model 5**: Ultimate STOCK-T (POM + EM + OVP)
3. Record the Strict and Relaxed Accuracies.

### 2. Aggregation and Output
After processing all specified trials, the script must compute the **mean** accuracy for each model across all trials.

### 3. Output (@docs)
* Print a Markdown table to the console showing the **Average Strict Accuracy** and **Average Relaxed Accuracy** for each of the 5 models.
* Save this robust, multi-trial ablation table to `docs/2026-05-02_Full_Scale_Ablation.md`.