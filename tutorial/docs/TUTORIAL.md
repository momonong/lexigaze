# IntelligentGaze: Neuro-Symbolic AI Workshop

## Overview
This tutorial demonstrates the implementation of a Neuro-Symbolic AI architecture. You will use a Large Language Model (LLM) to extract linguistic priors (Symbolic Cognition) to calibrate noisy gaze data from an edge eye-tracker (Neural Perception).

## Prerequisites
Python 3.10+ is required. Install the necessary dependencies:
```bash
pip install torch transformers pandas matplotlib seaborn numpy scipy flask python-dotenv
```

## Phase 1: Neural Perception (Data Collection)
Capture raw gaze data using a web-based eye-tracker.

1. Start the local data collection server:
   ```bash
   python tutorial/server.py
   ```
2. Open your browser to `http://localhost:8000`.
3. Allow camera access and follow the on-screen calibration steps.
4. Read the provided text. The system captures your gaze coordinates via WebGazer.
5. Export your dataset as `raw.csv` and ensure it is located in `tutorial/data/raw.csv`.

## Phase 2: Symbolic Cognition (Prior Extraction)
Use a BERT model to calculate "Word Surprisal" and determine gravitational weights for calibration.

1. Run the cognitive extraction script:
   ```bash
   python tutorial/text_model.py
   ```
   - This script uses `bert-tiny` to analyze word difficulty.
   - It calculates the Surprisal score for the target word "phenomenon".
   - It generates `tutorial/data/cognitive_weights.json` containing the calibration weight (Alpha).

## Phase 3: Neuro-Symbolic Fusion (Calibration)
Apply the symbolic prior to the raw neural perception data to correct hardware errors.

1. Run the calibration engine:
   ```bash
   python tutorial/calibrate.py
   ```
   - This script applies a Moving Average filter (Baseline).
   - It executes a "Gravity Snap" algorithm using the Alpha weight from Phase 2.
   - It produces `baseline.csv` and `calibrated.csv` in the `tutorial/data/` directory.

## Phase 4: Verification (Visual Analytics)
Compare the raw perception against the calibrated neuro-symbolic output.

1. Generate the comparison dashboard:
   ```bash
   python tutorial/heatmap.py
   ```
   - This script creates Kernel Density Estimation (KDE) heatmaps.
   - Review the results in `tutorial/figures/neuro_symbolic_dashboard.png`.

## Core Component Summary
- **server.py**: Local host for the WebGazer frontend.
- **text_model.py**: LLM-based linguistic surprisal calculator.
- **calibrate.py**: Fusion engine for perception and cognition.
- **heatmap.py**: Visualization tool for performance validation.
