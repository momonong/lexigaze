# Dataset Insights

This document summarizes the structure and characteristics of the datasets used in the LexiGaze project.

## 1. GECO Cleaned Trial Data (`data/geco/geco_pp01_trial5_clean.csv`)
This file contains the ground truth eye-tracking data for subject `pp01` reading `Trial 5`.
- **Columns**:
    - `WORD_ID`: Unique identifier for the word in the trial.
    - `WORD`: The literal word text.
    - `true_x`, `true_y`: The ground truth fixation coordinates (pixels).
    - `WORD_TOTAL_READING_TIME`: Duration of fixation on the word (ms).
- **Observations**: This data serves as the "Golden Standard" for evaluating our calibration algorithms.

## 2. Cognitive Mass Data (`data/geco/geco_pp01_cognitive_mass.csv`)
Likely contains the pre-calculated cognitive features for the words in the trial.
- **Columns**: `WORD_ID`, `WORD`, `true_x`, `true_y`, `surprisal_score`, `attention_score`, `cognitive_mass`.
- **Note**: Early inspection showed uniform scores (placeholder-like), suggesting a need for a fresh extraction pass using the BERT-based pipeline.

## 3. Bayesian Results (`data/geco/geco_pp01_bayesian_results.csv`)
Contains the output of the Static Bayesian Snap algorithm.
- **Columns**: Includes original fields plus `webcam_x/y` (simulated noise), `calibrated_x/y` (output), `snapped_word`, and accuracy flags.
- **Performance**: Provides a baseline for the upcoming Viterbi-based Spatio-Temporal decoder.

## 4. Raw Webcam Simulation (`tutorial/data/raw.csv`)
Simulated high-noise gaze coordinates typical of a laptop webcam.
- **Columns**: `timestamp`, `x_px`, `y_px`.
- **Characteristics**: Contains high-frequency jitter and systematic drift.

---
**Prepared by**: LexiGaze AI Orchestrator  
**Date**: May 1, 2026
