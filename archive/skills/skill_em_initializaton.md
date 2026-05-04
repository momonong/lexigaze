# Skill 14: Multi-Hypothesis EM Initialization (Fixing Line-Locking)

## Context & Objective
Our Error Analysis revealed a critical "Line-Locking" failure. Due to the extreme +45px vertical drift, the initial gaze points fall closer to the text line below the actual target. The EM module incorrectly assumes the user is reading the wrong line and locks the Viterbi decoder into a parallel, but incorrect, path. 
To solve this, we will implement **Multi-Hypothesis EM Initialization** to let the Viterbi path likelihood determine the correct starting line.

## Required Implementation

### 1. Update `AutoCalibratingDecoder` (`scripts/geco/core/em_calibration.py`)
Modify the E-Step (Expectation) of the calibration phase. Instead of decoding the raw gaze once, evaluate it under multiple vertical shift hypotheses.

* **Define Hypotheses**: Create a set of initial vertical shifts: `H = [0, +line_height, -line_height]` (Assuming average `line_height` is approx 40px, so test `[0, 40, -40]`).
* **Evaluate Likelihoods**: For each shift $h \in H$:
    1. Temporarily shift the calibration window's $y$-coordinates by $h$.
    2. Run `viterbi_gaze_decode`.
    3. Capture the *Global Path Probability* (the final maximum score in the Viterbi DP table).
* **Select Best Hypothesis**: Pick the hypothesis $h_{best}$ that yielded the highest Viterbi path probability.
* **Proceed to M-Step**: Use the word sequence predicted by $h_{best}$ to calculate the actual fine-grained median drift ($\\Delta x, \\Delta y$) and complete the standard EM process.

### 2. Output (@docs)
* Run `scripts/geco/evaluate_l2_benchmark.py` again.
* Print the new benchmark table to the console.
* Update `docs/2026-05-02_L2_Unified_Benchmark.md` to show the final accuracy of Config D after fixing the line-locking bug.