# Skill 22: Comprehensive NeurIPS Synthesis (The "Grand Finale" Draft)

## Context & Objective
We have moved beyond individual experiments. We now have a full suite of theoretical proofs, technical logs, and population-level statistical evidence. 
The goal is to perform a **Global Synthesis** of the entire `docs/` folder to produce a v2 research paper draft. This version must be ready for LaTeX conversion and final submission.

## Processing Requirements
1. **Recursive Context Gathering**: Read and synthesize ALL files in `docs/` and its subdirectories (`experiments/`, `figures/`).
2. **Directory Structure**: Create a new folder `docs/NeurIPS/` for all final paper-related artifacts.

## Required Outputs (@docs/NeurIPS/)

### 1. The English Manuscript (`docs/NeurIPS/manuscript_v2_en.md`)
A full 8-9 page structured paper including:
- **Title**: Refined for maximum impact (Neuro-Symbolic / Edge AI / Robustness).
- **Abstract**: Highlighting the 86.4% L2 average accuracy and 99.6% semantic recovery under extreme drift.
- **Introduction**: Unified narrative from `PROJECT.md` and `DATASET_INSIGHTS.md`.
- **Methodology**: 
    - Formalize **Cognitive Mass** ($CM = S \times A$).
    - Detail the **POM Transition Matrix** logic.
    - Detail **Multi-Hypothesis EM** for breaking Line-Locking.
- **Experimental Results**:
    - Table: Full Corpus Benchmark (L1 vs. L2).
    - Table: Noise Stress Test results (0px to 75px).
    - Qualitative Analysis: Description of the decoded path success.
- **Discussion**: The "Proficiency-Adaptive OVP" discovery (The core scientific contribution).

### 2. The Traditional Chinese Version (`docs/NeurIPS/manuscript_v2_ch.md`)
A precise translation of the above, used for final alignment with the advisor.

### 3. LaTeX Helper Metadata (`docs/NeurIPS/latex_refs.md`)
Extract all key mathematical formulas, table data (in CSV or LaTeX format), and figure captions into one file to speed up Overleaf entry.

## Formatting Tone
- Academic, rigorous, and defensive (NeurIPS Standard).
- Ensure all statistical terms (p-values, SD, Mean) are correctly placed.