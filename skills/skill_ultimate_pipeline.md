# Skill 11 (Final Final): The Unified L2 Benchmark & Hyperparameter Grid Search

## Context & Objective
The lead researcher verified the dataset: subject `pp01` is indeed a bilingual (L2) reader. However, our new Psycholinguistic-Oculomotor Model (POM) achieved only 46.2% accuracy compared to the Base Viterbi (50.0%). 
Since the subject is correct, the performance drop is due to unoptimized POM hyperparameters ($\\sigma_{fwd}$, $\\sigma_{reg}$, $\\gamma$). 
To finalize our NeurIPS paper, we must run a unified benchmark evaluating all baselines against a formally hyperparameter-tuned POM model on this exact L2 dataset.

## Required Implementation

### 1. Unified Evaluation Script (`scripts/geco/evaluate_l2_benchmark.py`)
Create a master script that loads the L2 dataset (`L2ReadingData.xlsx`, trial 5 of `pp01`) and evaluates it using four configurations:
* **Config A (Spatial Baseline)**: `NearestBoundingBoxDecoder`
* **Config B (Temporal Baseline)**: `StandardKalmanDecoder`
* **Config C (Base Viterbi)**: Standard Viterbi (Rule-based transitions, no POM).
* **Config D (STOCK-T / Ultimate LexiGaze)**: 
    * `DynamicCognitiveField` (OVP shift at 35% width).
    * `PsycholinguisticTransitionMatrix` (POM).
    * `AutoCalibratingDecoder` (EM Drift Correction).

### 2. Hyperparameter Grid Search (For Config D Only)
Before running Config D in the final benchmark, implement a grid search to automatically find the parameter set that maximizes Word-Level Fixation Accuracy:
* `sigma_fwd`: [0.8, 1.0, 1.2, 1.5] (Forward saccade variance)
* `sigma_reg`: [1.5, 2.0, 3.0] (Regression variance)
* `gamma`: [0.3, 0.5, 0.7] (Cognitive penalty weight)

### 3. Output & Documentation (@docs)
1.  Print the optimal parameters found for POM.
2.  Print a Markdown-formatted table to the terminal comparing the Word-Level Fixation Accuracy (%) of Config A, B, C, and the optimized Config D.
3.  Save this comparative table into `docs/2026-05-02_L2_Unified_Benchmark.md`. This will serve as "Table 1" in our publication.