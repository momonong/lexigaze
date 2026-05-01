# Skill 12: Dual-Metric Evaluation (Strict vs. Relaxed ROI)

## Context & Objective
For our NeurIPS submission, we need to report our results using two metrics. Human foveal vision spans approximately 3-4 words, meaning readers process adjacent words simultaneously (parafoveal preview). Therefore, evaluating eye-tracking strictly by exact word match under-represents the system's true semantic tracking capability.
We will implement a "Relaxed Accuracy" metric ($\\pm 1$ word tolerance) alongside our existing "Strict Accuracy".

## Required Implementation

### 1. Update Evaluation Logic (`scripts/geco/evaluate_l2_benchmark.py`)
Modify the accuracy calculation function to compute two scores for every configuration (Config A, B, C, and D):
* **Strict Accuracy**: Predicted Word Index == Ground Truth Word Index.
* **Relaxed Accuracy ($\\pm 1$ Word)**: `abs(Predicted Word Index - Ground Truth Word Index) <= 1`.

*Note: Ensure bounds checking so that the first and last words of a sentence don't cause index out-of-bounds errors.*

### 2. Update Console Output & Markdown Report
Modify the output table to include both metrics. The table in `docs/2026-05-02_L2_Unified_Benchmark.md` should now look like this:

| Configuration | Strict Accuracy (%) | Relaxed Accuracy ($\\pm 1$) (%) |
| :--- | :---: | :---: |
| Config A: Nearest Bounding Box | ... | ... |
| Config B: Standard Kalman Filter | ... | ... |
| Config C: Base Viterbi (Rule-based) | ... | ... |
| Config D: Ultimate STOCK-T (POM) | ... | ... |

## Output (@docs)
* Run the benchmark again with the optimal parameters found in Skill 11 (`sigma_fwd=0.8, sigma_reg=1.5, gamma=0.3`).
* Update the markdown report with the new Dual-Metric table. 
* Add a brief discussion in the report explaining that the Relaxed Accuracy accounts for the biological ~2° foveal visual angle.