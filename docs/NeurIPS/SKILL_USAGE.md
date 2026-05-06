# Skill Guide: Neuro-Symbolic Gaze Calibration

This document explains how to use the **Cognitive Mass (CM)** skill and the reorganized scripts in the `scripts/geco/` directory.

## 1. The "Cognitive Mass" Skill
The core objective of this skill is to fuse **Neural Perception** (noisy webcam gaze data) with **Symbolic Cognition** (linguistic importance of words) to improve eye-tracking accuracy on edge devices.

- **Neural Component**: Large Language Models (BERT) calculate "Surprisal" and "Attention" to determine how "heavy" a word is cognitively.
- **Symbolic Component**: Linguistic rules (Dependency Parsing, Age of Acquisition) provide a stable structural prior.
- **Result**: A "Cognitive Mass" (CM) score between 0 and 1 for every word.

## 2. Reorganized Script Structure
The GECO (Ghent Eye-Tracking Corpus) scripts are now organized into four functional categories:

### 📂 `scripts/geco/core/`
Contains the mathematical and algorithmic engines.
- **`cm_algorithm.py`**: The main implementation of the `CognitiveMassCalculator` and `BayesianGravitySnap` classes.
- **`gravity_snap.py`**: An alternative "Event Horizon" logic based on Sigmoid-weighted radii.

### 📂 `scripts/geco/data/`
Utilities for handling raw experimental data.
- **`data_loader.py`**: A robust loader for the GECO L2 Excel database.
- **`check_data.py`**: A quick verification tool to inspect raw data columns.
- **`extract_sample.py`**: Cleans and filters the massive GECO dataset into lightweight CSV files for experimentation.

### 📂 `scripts/geco/demo/`
End-to-end demonstrations of the skill.
- **`demo_bayesian_snap.py`**: The primary demo showing how CM corrects vertical drift and webcam jitter.

### 📂 `scripts/geco/tasks/`
Production-ready pipeline tasks.
- **`extract_cognitive_mass.py`**: A batch processing script to compute CM for any given dataset.

## 3. Standard Workflow

### Step 1: Prepare Data
Extract a specific trial (e.g., Subject pp01, Trial 5) to use as ground truth:
```bash
python scripts/geco/data/extract_sample.py
```

### Step 2: Extract Cognitive Features
Run the BERT-based pipeline to compute the "gravity" of each word:
```bash
python scripts/geco/tasks/extract_cognitive_mass.py
```

### Step 3: Run Calibration Demo
Simulate a noisy webcam and see the Bayesian fusion correct the gaze:
```bash
python scripts/geco/demo/demo_bayesian_snap.py
```

---
*For more technical details on the CM formulas, refer to the [Cognitive Mass Theory](./COGNITIVE_MASS.md) or the [Skill Definition](../skills/skill_cognitive_mass.md).*
