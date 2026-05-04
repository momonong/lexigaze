# Skill 6: EM-based Dynamic Drift Auto-Calibration

## Context & Objective
We are upgrading our Spatio-Temporal Viterbi Decoder to perform Zero-Shot Auto-Calibration. In Edge AI eye-tracking, hardware drift (e.g., +45px vertical offset due to head tilt or poor webcam calibration) is the biggest point of failure.
We will use an Expectation-Maximization (EM) inspired approach: Use the cognitive reading model to guess what the user *intended* to look at, calculate the systematic hardware error, and correct the sensor data on the fly.

## The Algorithm (EM-Viterbi)
Create a class `AutoCalibratingDecoder` in a new file `em_calibration.py`.

### Step 1: E-Step (Expectation)
1. Take the first $K$ frames of gaze data (e.g., the first 1.5 seconds of a sentence reading trial).
2. Run our existing `viterbi_gaze_decode` (from Skill 3) on this initial chunk to get the most likely sequence of intended word targets.

### Step 2: M-Step (Maximization / Drift Estimation)
1. For the $K$ frames, compute the error vector between the raw gaze coordinates and the actual center coordinates of the Viterbi-predicted words.
2. Calculate the robust mean drift vector:
   $$\Delta x = \text{Median}(gaze\_x[1..K] - predicted\_word\_center\_x[1..K])$$
   $$\Delta y = \text{Median}(gaze\_y[1..K] - predicted\_word\_center\_y[1..K])$$
   *(Note: Using Median instead of Mean to resist outlier saccades/blinks).*

### Step 3: Update & Final Decode
1. Apply the drift correction to the *entire* trial's raw gaze sequence: 
   $corrected\_gaze = raw\_gaze - (\Delta x, \Delta y)$
2. Re-run the `viterbi_gaze_decode` on the `corrected_gaze` to get the final, highly accurate reading path.

## Required Implementation
* Implement `calibrate_and_decode(raw_gaze_sequence, word_boxes, base_cm, transition_matrix, calibration_window_size)`.
* Include logic to handle edge cases (e.g., if the user wasn't reading during the first $K$ frames, indicated by a massive variance in the error vector).