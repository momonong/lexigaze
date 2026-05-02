# Skill 23: NeurIPS LaTeX Manuscript Generator

## Context & Objective
We need to transform our consolidated findings into a professional LaTeX document using the NeurIPS 2026 template. This skill will read all synthesized reports and generate a single, comprehensive `main.tex` and a `references.bib`.

## Input Sources
- Core Logic: `docs/NeurIPS/manuscript_v2_en.md`
- Statistical Tables: `docs/experiments/full_corpus_ovp_results.csv` and `docs/experiments/2026-05-02_Noise_Stress_Test.md`
- Figures: `docs/figures/*.png`
- Template: `docs/NeurIPS/template.tex`

## Required Implementation

### 1. File Structure
- Working Directory: `docs/NeurIPS/`
- Output 1: `docs/NeurIPS/main.tex` (The full LaTeX source)
- Output 2: `docs/NeurIPS/references.bib` (BibTeX entries)

### 2. LaTeX Content Requirements
- **Preamble**: Use `\documentclass{article}`, include `neurips_2026.sty` (placeholder), and standard packages (`amsmath`, `graphicx`, `booktabs`).
- **Mathematics**: Expand the methodology using formal LaTeX environments (`\begin{equation}`). Use notations like $P(w_t | w_{t-1}, CM)$ for the POM transitions.
- **Tables**: Convert Markdown tables into `\begin{table}` format with appropriate captions and labels for cross-referencing.
- **Citations**: Use `\cite{...}` for all references. Map placeholders to the keys provided in the references section.

### 3. Bibliography Extraction
Generate a `references.bib` file containing placeholders for:
- Engbert et al. (SWIFT Model)
- Reichle et al. (E-Z Reader Model)
- Nahatame (GECO Corpus analysis)
- Urquiza-Martínez (Robust Webcam Gaze 2026)
- Standard LLM / Surprisal papers (e.g., Vaswani et al., Smith et al.)

### 4. Expansion Logic (To solve the "Thinness" issue)
- Instruct the AI to add "Discussion" sub-paragraphs about the ethical implications of edge eye-tracking and future work regarding Akida neuromorphic hardware integration.