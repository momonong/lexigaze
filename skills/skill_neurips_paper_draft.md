# Skill 17: NeurIPS Paper Draft Architect (Neuro-Symbolic Gaze)

## Context & Objective
We have completed our algorithmic development and ablation studies. Our pipeline (POM + Multi-Hypothesis EM) achieved 90.49% average strict accuracy under extreme +45px drift, effectively solving the "Line-Locking" failure.
The goal is to synthesize all documents in the `/docs` folder into a structured research paper draft. This draft will be used to discuss with our advisor and eventually be converted to LaTeX.

## Input Context (Please read these files)
1. `docs/PROJECT.md`: Project vision and Cognitive Mass formula.
2. `docs/2026-05-02_Full_Scale_Ablation.md`: The multi-trial data (90%+ results).
3. `docs/2026-05-02_Error_Analysis.md`: The "Line-Locking" pathology and "OVP Fail" insight.
4. `docs/SKILL_IMPLEMENTATION_LOG.md`: The technical evolution steps.

## Required Output Structure (Bilingual: English & Traditional Chinese)
Generate a Markdown document `docs/neurips_draft_v1.md` containing:

1. **Title**: A professional NeurIPS-style title (e.g., "Overcoming Extreme Baseline Drift in Edge Eye-Tracking via Neuro-Symbolic Cognitive Modeling").
2. **Abstract**: Emphasize the +45px drift challenge and the 90%+ recovery.
3. **Introduction**: Frame the problem of low-cost hardware vs. cognitive intent.
4. **Methodology**: 
    - Detail the **Cognitive Mass (Surprisal x Attention)**.
    - Detail the **Psycholinguistic-Oculomotor Model (POM)** transitions.
    - Detail the **Multi-Hypothesis EM Calibration** (The breakthrough for line-locking).
5. **Experiments & Results**:
    - Include the **Ablation Table** from `Full_Scale_Ablation.md`.
    - **Crucial Discussion Section**: Explain the "OVP Anomaly" – why L2 readers prefer the geometric center over the biological OVP.
6. **Future Work**: Mention Knowledge Distillation to Edge devices and Adaptive OVP.

## Formatting Rules
- Use rigorous academic tone.
- Ensure mathematical formulas (LaTeX style) are correctly transcribed.
- Provide Traditional Chinese translation immediately following each English section.