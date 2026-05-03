# Reproducibility Reset Report (2026-05-03)

## 1. Why This Reset Was Done

This reset was executed to eliminate data lineage ambiguity between:

- legacy demo outputs (for early Bayesian visualization),
- current STOCK-T evaluation scripts,
- and NeurIPS figure-generation scripts.

The goal is to ensure all publication-facing artifacts are tied to explicitly rerun scripts and saved outputs.

## 2. Scripts Re-run in This Pass

The following experiment scripts were re-executed on the current workspace state:

- `scripts/geco/tasks/evaluate_pipeline.py`
- `scripts/geco/evaluate_l2_benchmark.py`
- `scripts/geco/evaluate_ablation.py`
- `scripts/geco/evaluate_population.py`
- `scripts/geco/evaluate_noise_stress.py`

The following figure scripts were re-executed:

- `scripts/geco/generate_neurips_figures_v2.py`
- `scripts/geco/generate_neurips_figures_final.py`

## 3. Verified Current Outputs (Post-Rerun)

### 3.1 pp01 Trial 5 Core Evaluation

From `data/geco/geco_pp01_final_evaluation.csv`:

- Nearest Box: 18.59%
- Kalman Filter: 2.56%
- Static Bayesian: 16.67%
- Viterbi (Base): 49.36%
- Viterbi + OVP: 49.36%
- Viterbi + EM-AutoCal: 73.72%
- STOCK-T v1: 37.82%
- STOCK-T v2: 33.33%
- STOCK-T v3 (POM): 78.21%

### 3.2 Unified L2 Benchmark (Grid Search Config D)

From `docs/2026-05-02_L2_Unified_Benchmark.md`:

- Config A: 18.59% / 26.92% (strict/relaxed)
- Config B: 2.56% / 5.77%
- Config C: 48.72% / 61.54%
- Config D (Ultimate STOCK-T): 78.85% / 98.72%

### 3.3 Ablation (pp01 Trial 5)

From `docs/2026-05-02_NeurIPS_Ablation_Study.md`:

- M1 Base Viterbi: 48.72% / 61.54%
- M2 + Multi-Hypothesis EM: 77.56% / 98.72%
- M3 + EM + OVP (No POM): 73.72% / 98.72%
- M4 STOCK-T (POM + EM, No OVP): 83.33% / 100.00%
- M5 Ultimate STOCK-T (POM + EM + OVP): 78.85% / 98.72%

### 3.4 Noise Stress Test

From `data/geco/noise_stress_results.csv`:

- Drift 0: Baseline 32.34%, EM Only 81.54%, STOCK-T 90.49%
- Drift 45: Baseline 19.10%, EM Only 74.90%, STOCK-T 90.49%
- Drift 75: Baseline 8.50%, EM Only 54.86%, STOCK-T 51.95%

### 3.5 Cross-Subject OVP Analysis

From `docs/experiments/2026-05-02_Cross_Subject_OVP_Analysis.md`:

- L1 mean: Center 93.26%, OVP 90.91%, diff -2.34%, p=0.2837
- L2 mean: Center 86.71%, OVP 85.35%, diff -1.36%, p=0.2299

## 4. Figure 4 Data Lineage (Critical)

Figure 4 in `scripts/geco/generate_neurips_figures_final.py` now reads:

- `data/geco/geco_pp01_trial5_stockt_from_main_m4_trajectory.csv`

This replaces legacy dependency on:

- `data/geco/geco_pp01_bayesian_results.csv`

Therefore, Figure 4 is now connected to re-computed STOCK-T trajectory output rather than early Bayesian demo output.

## 5. Newly Persisted Trajectory Artifacts

Saved for reproducible scanpath rendering:

- `data/geco/geco_pp01_trial5_stockt_from_main_m4_trajectory.csv`
- `data/geco/geco_pp01_trial5_stockt_from_main_m5_trajectory.csv`
- `data/geco/geco_pp01_trial5_stockt_from_main_metrics.csv`
- `data/geco/geco_pp01_trial5_stockt_from_population_m4_trajectory.csv`
- `data/geco/geco_pp01_trial5_stockt_from_population_m5_trajectory.csv`
- `data/geco/geco_pp01_trial5_stockt_from_population_metrics.csv`

## 6. Integrity Assessment

Current evidence does not indicate fabricated results. The risk observed was version drift across scripts/reports/data products, not fake data generation in final artifacts.

The main integrity risk was mixing:

- old demo outputs,
- newer model evaluation outputs,
- and static values in figure scripts.

This reset significantly reduces that risk by rerunning scripts and relinking Figure 4 to persisted STOCK-T trajectories.

## 7. Remaining Recommendations Before Submission

- Freeze one canonical experiment snapshot for paper numbers (single rerun tag/date).
- Add explicit "Data Source" lines below each NeurIPS figure caption.
- Keep `generate_neurips_figures_v2.py` and `generate_neurips_figures_final.py` aligned to the same refreshed metric sources.
- If final manuscript cites full-corpus metrics, rerun corresponding full-corpus pipeline once and lock the output CSV hash/date.

---

Prepared by: LexiGaze Reproducibility Reset  
Date: 2026-05-03