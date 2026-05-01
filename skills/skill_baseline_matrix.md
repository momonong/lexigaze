# Skill 5: NeurIPS Baseline Decoders Implementation

## Context & Objective
You are an AI researcher preparing the ablation study for a NeurIPS paper on the LexiGaze eye-tracking system. 
To prove the superiority of our Spatio-Temporal Viterbi Decoder (STOCK-T), we need to implement three strict baselines. This will demonstrate that without BOTH cognitive priors and temporal dynamics, extreme hardware drift (+45px) cannot be resolved.

## Required Implementation
Create a module `baseline_decoders.py` containing three distinct decoder classes. Each class should have a `decode(gaze_sequence, word_boxes, base_cm)` method that returns the predicted sequence of fixated words.

### 1. NearestBoundingBoxDecoder (Pure Spatial Heuristic)
* **Logic**: For each gaze point $(x_t, y_t)$, find the word bounding box whose center is geometrically closest (Euclidean distance).
* **Purpose**: Proves that raw proximity fails under systematic hardware drift.

### 2. StandardKalmanDecoder (Temporal, No Cognitive Prior)
* **Logic**: 
  1. Apply a standard 2D Constant Velocity Kalman Filter to smooth the `gaze_sequence` to remove high-frequency jitter.
  2. Map the smoothed trajectory to words using the Nearest Bounding Box method.
* **Purpose**: Proves that purely physical trajectory smoothing (standard CV tracking) is insufficient without linguistic awareness.

### 3. StaticBayesianDecoder (Cognitive, No Temporal)
* **Logic**: Implements our legacy v1 approach.
  For each gaze point $g_t$, calculate the probability for each word $w_i$:
  $$P(w_i | g_t) \propto \mathcal{N}(g_t | \mu_i, \Sigma) \times \text{Base\_CM}_i$$
  Choose the word with the $\max$ probability independently for each time step.
* **Purpose**: Proves that a static cognitive prior gets "stuck" on difficult words and fails to model the forward momentum of reading.

## Output
Provide clean, vectorized Python code using NumPy. Ensure these decoders share a common interface with our new Viterbi decoder so they can be easily swapped in our evaluation pipeline (Skill 4).