# Skill 15: NeurIPS Ablation Study (Component Isolation)

## Context & Objective
The Multi-Hypothesis EM successfully pushed the accuracy to 92.31%. However, for a rigorous NeurIPS paper, we must conduct an Ablation Study to prove that *each* component (EM, POM, OVP) independently contributes to this final performance, rather than the EM module doing all the work alone.

We need to evaluate a staircase of models on the L2 dataset (Trial 5) using the Dual-Metric system (Strict & Relaxed).

## Required Implementation

### 1. The Ablation Script (`scripts/geco/evaluate_ablation.py`)
Create a script that evaluates the following 5 specific configurations on the same +45px drifted data:

* **Model 1: Base Viterbi (No EM, No POM, No OVP)**
  - Rule-based transitions, center-bounding box emissions, single-pass decode.
* **Model 2: Viterbi + Multi-Hypothesis EM (No POM, No OVP)**
  - Proves the power of the dynamic drift calibration.
* **Model 3: Viterbi + Multi-Hypothesis EM + OVP (No POM)**
  - Proves that shifting the target to the biological optimal viewing position (35% width) improves spatial likelihoods.
* **Model 4: STOCK-T (POM + Multi-Hypothesis EM) (No OVP)**
  - Replaces rule-based transitions with the Psycholinguistic-Oculomotor Model. Proves that cognitive mass and surprisal improve the sequence tracking.
* **Model 5: Ultimate STOCK-T (POM + Multi-Hypothesis EM + OVP)**
  - The complete architecture.

### 2. Hyperparameter Locking
For Models 4 and 5, strictly use the optimal parameters found earlier: `sigma_fwd=0.8, sigma_reg=1.5, gamma=0.3`.

### 3. Output (@docs)
* Print the Ablation Table (5 Models x 2 Metrics) to the console.
* Save the exact output to `docs/2026-05-02_NeurIPS_Ablation_Study.md`.