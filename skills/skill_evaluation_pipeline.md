# Skill 4: NeurIPS Evaluation & Ablation Pipeline

## Context & Objective
We need to rigorously evaluate our new Spatio-Temporal Viterbi Decoder against our baseline (Static Bayesian Snap) using the GECO dataset. 
The primary metric is "Word-Level Fixation Accuracy" under simulated extreme hardware noise (+45px vertical/horizontal drift).

## Required Implementation
Create a script `evaluate_pipeline.py`:
1.  **Noise Injection**: Create a function to inject a systematic $+45$px drift and Gaussian jitter to the ground-truth GECO gaze coordinates.
2.  **Ablation Configurations**: Run the dataset through 3 configurations:
    * Config A: Raw proximity matching (Nearest Bounding Box).
    * Config B: Static Bayesian Snap (Baseline).
    * Config C: Spatio-Temporal Viterbi Decoder (Ours).
3.  **L1 vs L2 Segregation**: Calculate metrics separately for L1 (native) and L2 (bilingual) subjects in the GECO dataset.
4.  **Metrics**: 
    * Accuracy (%).
    * Mean Distance Error (px).

## Output
Generate a well-formatted Pandas DataFrame and save a comparative Bar Chart using Matplotlib/Seaborn. Ensure the styling is academic (high DPI, clear legends) suitable for a NeurIPS paper.