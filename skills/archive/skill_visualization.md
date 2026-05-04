# Skill 13: Error Analysis & Gaze Path Visualization

## Context & Objective
We have achieved ~50% Strict and ~58% Relaxed accuracy. To push the algorithm further without blindly overfitting (P-hacking), we need to perform rigorous Error Analysis. 
We must visualize the "Crime Scene" to understand EXACTLY where our Spatio-Temporal Viterbi decoder is failing (e.g., line breaks, short words, long regressions).

## Required Implementation

### 1. Visualization Script (`scripts/geco/error_analysis_visualizer.py`)
Create a Python script using `matplotlib` (or `plotly` for interactive HTML) to visualize a specific sequence of the reading trial.
The plot must include:
1.  **Word Bounding Boxes**: Draw rectangles for each word in the sentence. Overlay the actual text of the word inside or near the box.
2.  **Raw Gaze Points**: Plot the raw, noisy gaze coordinates (with the +45px drift) as scatter points (e.g., faint red dots).
3.  **Ground Truth Path**: Plot the actual sequence of words the user looked at (e.g., solid green line connecting the centers of the ground-truth words).
4.  **STOCK-T Predicted Path**: Plot the path decoded by our optimal Config D (POM + EM + OVP) (e.g., dashed blue line).

### 2. Failure Logging
Implement a console output that lists the specific words where the prediction failed. 
Format: `Time t | True Word: "apple" (idx: 5) | Predicted Word: "the" (idx: 4) | Note: Relaxed Match / Complete Miss`

## Output (@docs)
* Run this script on the first 50-100 fixations of `pp01` Trial 5.
* Save the output plot as an image to `docs/figures/trial5_analysis.png`.
* Generate a brief markdown report `2026-05-02_Error_Analysis.md` identifying any noticeable patterns in the failures (e.g., "Failing at line breaks", "Getting stuck on short words").