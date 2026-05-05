import os
import sys
import pandas as pd
import numpy as np
import traceback
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns

# 確保專案根目錄正確
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.geco.core.transition_model import PsycholinguisticTransitionMatrix
from scripts.geco.core.em_calibration import AutoCalibratingDecoder
from scripts.geco.core.baseline_decoders import NearestBoundingBoxDecoder
from scripts.geco.core.geco_metrics import evaluate_word_and_recovery, stable_seed, word_line_ids_from_layout

# 參數設定
SIGMA_FWD = 0.8
SIGMA_REG = 1.5
GAMMA = 0.3
SIGMA_X = 40.0
SIGMA_Y = 30.0

# 測試不同的垂直漂移強度 (從完美的 0px 到極端的 60px)
DRIFT_LEVELS = [0.0, 15.0, 30.0, 45.0, 60.0]

def decode_nearest_y_only(gaze_sequence: np.ndarray, word_boxes: list[list[float]]) -> list[int]:
    """
    A *strict* spatial baseline that ignores X and snaps by vertical proximity only.
    This matches the physical intuition behind "line-locking": with systematic vertical drift,
    a purely spatial Y-based decoder should collapse to the wrong line.
    """
    word_centers_y = np.array([(box[1] + box[3]) / 2 for box in word_boxes], dtype=float)
    preds: list[int] = []
    for gaze in gaze_sequence:
        if np.isnan(gaze).any():
            preds.append(preds[-1] if preds else 0)
            continue
        dy2 = (word_centers_y - float(gaze[1])) ** 2
        preds.append(int(np.argmin(dy2)))
    return preds

def inject_noise(df, drift_y, rng):
    df['true_x'] = pd.to_numeric(df['true_x'], errors='coerce')
    df['true_y'] = pd.to_numeric(df['true_y'], errors='coerce')
    df = df.dropna(subset=['true_x', 'true_y']).copy()

    n = len(df)
    df['noisy_x'] = df['true_x'] + rng.normal(0, SIGMA_X, n)
    df['noisy_y'] = df['true_y'] + rng.normal(0, SIGMA_Y, n) + drift_y
    return df

def _collect_tasks(
    *,
    limit_subjects_per_lang: int | None = None,
    limit_trials_per_subject: int | None = None,
    drift_levels: list[float] | None = None,
) -> list[tuple]:
    drift_levels = drift_levels or DRIFT_LEVELS
    tasks: list[tuple] = []
    for lang in ["L1", "L2"]:
        pop_dir = os.path.join(PROJECT_ROOT, f"data/geco/population/{lang}")
        if not os.path.exists(pop_dir):
            continue

        subjects = [s for s in os.listdir(pop_dir) if os.path.isdir(os.path.join(pop_dir, s))]
        subjects = sorted(subjects)
        if limit_subjects_per_lang is not None:
            subjects = subjects[: max(0, int(limit_subjects_per_lang))]

        for sub in subjects:
            sub_dir = os.path.join(pop_dir, sub)
            trials = [d for d in os.listdir(sub_dir) if d.startswith("trial_")]
            trials = sorted(trials)
            if limit_trials_per_subject is not None:
                trials = trials[: max(0, int(limit_trials_per_subject))]

            for trial in trials:
                layout_path = os.path.join(sub_dir, trial, "layout.csv")
                fixations_path = os.path.join(sub_dir, trial, "fixations.csv")
                if os.path.exists(layout_path) and os.path.exists(fixations_path):
                    for drift in drift_levels:
                        tasks.append((lang, sub, trial, layout_path, fixations_path, drift))
    return tasks

def _sanity_check_noise(df_fixations: pd.DataFrame, drift_y: float) -> dict:
    """
    Quick checks to confirm we are using noisy coords and drift is applied.
    Returns summary stats for printing.
    """
    if "true_y" not in df_fixations.columns or "noisy_y" not in df_fixations.columns:
        return {"ok": False, "reason": "missing true_y/noisy_y"}
    dy = (pd.to_numeric(df_fixations["noisy_y"], errors="coerce") - pd.to_numeric(df_fixations["true_y"], errors="coerce"))
    dy = dy.dropna()
    if dy.empty:
        return {"ok": False, "reason": "dy empty"}
    return {
        "ok": True,
        "drift_y": float(drift_y),
        "dy_mean": float(dy.mean()),
        "dy_median": float(dy.median()),
        "dy_std": float(dy.std()),
        "n_points": int(len(dy)),
    }

def run_noise_tolerance_experiment(
    *,
    test_mode: bool = False,
    limit_subjects_per_lang: int | None = None,
    limit_trials_per_subject: int | None = None,
):
    benchmark_dir = os.path.join(PROJECT_ROOT, "data", "geco", "benchmark")
    os.makedirs(benchmark_dir, exist_ok=True)

    tasks = _collect_tasks(
        limit_subjects_per_lang=limit_subjects_per_lang if test_mode else None,
        limit_trials_per_subject=limit_trials_per_subject if test_mode else None,
        drift_levels=DRIFT_LEVELS,
    )

    print(f"Total experiment runs scheduled: {len(tasks)} (Trials x Drift Levels)")
    if test_mode:
        # Distribution preview
        langs = [t[0] for t in tasks]
        drifts = [t[5] for t in tasks]
        print(f"[TEST] Lang counts: L1={langs.count('L1')}, L2={langs.count('L2')}")
        print(f"[TEST] Drift counts: " + ", ".join(f"{d:.0f}px={drifts.count(d)}" for d in DRIFT_LEVELS))

    all_results = []

    if test_mode:
        # Run sequentially for fast, debuggable sanity checks
        for task in tqdm(tasks, desc="[TEST] Running small drift set"):
            lang, sub, trial, layout_path, fixations_path, drift_y = task

            # Pre-check: confirm noisy drift is being applied as expected
            rng = np.random.default_rng(stable_seed(lang, sub, trial, drift_y))
            df_fix = pd.read_csv(fixations_path).rename(columns={"fixation_x": "true_x", "fixation_y": "true_y"})
            df_fix_n = inject_noise(df_fix, drift_y, rng)
            sc = _sanity_check_noise(df_fix_n, drift_y)
            if sc.get("ok"):
                print(
                    f"[TEST] {lang}/{sub}/{trial} drift={drift_y:.0f}px: Δy mean={sc['dy_mean']:.2f}, "
                    f"median={sc['dy_median']:.2f}, std={sc['dy_std']:.2f} (n={sc['n_points']})"
                )
            else:
                print(f"[TEST] {lang}/{sub}/{trial} drift={drift_y:.0f}px: sanity check failed: {sc}")

            res = process_single_trial_with_drift(task)
            if res:
                all_results.append(res)
    else:
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(process_single_trial_with_drift, task): task for task in tasks}
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Running Drift Stress Test"):
                res = future.result()
                if res: all_results.append(res)

    df_results = pd.DataFrame(all_results)
    csv_path = os.path.join(benchmark_dir, "noise_tolerance_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nExperiment finished. Data saved to {csv_path}")

    if test_mode:
        print("[TEST] Result columns: " + ", ".join(df_results.columns.tolist()))
        print("[TEST] Head:")
        print(df_results.head(5).to_string(index=False))
        # In test mode, do not auto-plot by default (fast feedback)
        return

    # 直接畫圖並存到 docs\NeurIPS\figures
    plot_noise_tolerance(df_results)

def process_single_trial_with_drift(args):
    lang, sub, trial, layout_path, fixations_path, drift_y = args
    rng = np.random.default_rng(stable_seed(lang, sub, trial, drift_y))

    try:
        df_layout = pd.read_csv(layout_path)
        df_fixations = pd.read_csv(fixations_path)
        
        if df_layout.empty or df_fixations.empty: return None
            
        df_fixations = df_fixations.rename(columns={'fixation_x': 'true_x', 'fixation_y': 'true_y'})
        df_fixations = inject_noise(df_fixations, drift_y, rng)
        if df_fixations.empty: return None

        line_by_word = word_line_ids_from_layout(df_layout)

        cm_raw = df_layout['cognitive_mass'].values
        cm_real = pd.Series(cm_raw).rolling(window=3, center=True, min_periods=1).mean().values
        cm_uniform = np.ones(len(df_layout)) * 2.5
        
        word_boxes = []
        for _, row in df_layout.iterrows():
            word_str = str(row['WORD']).strip()
            w = max(40.0, len(word_str) * 12.0)
            h = 40.0
            word_boxes.append([
                row['true_x'] - w/2, row['true_y'] - h/2, 
                row['true_x'] + w/2, row['true_y'] + h/2
            ])
        
        gaze_seq = df_fixations[['noisy_x', 'noisy_y']].values
        targets = df_fixations['layout_index'].values.astype(int)
        cal = AutoCalibratingDecoder()
        
        # 1. STOCK-T_Edge (Uniform CM + POM)
        t_pom = PsycholinguisticTransitionMatrix(sigma_fwd=SIGMA_FWD, sigma_reg=SIGMA_REG, gamma=GAMMA).build_matrix(len(df_layout), cm_real)
        idx, drift = cal.calibrate_and_decode(gaze_seq, word_boxes, cm_uniform, t_pom, use_ovp=True)
        _, _, edge_rec, _ = evaluate_word_and_recovery(targets, idx, line_by_word, drift[1], drift_y)

        # 2. STOCK-T_Surprisal (Real CM + POM)
        idx, drift = cal.calibrate_and_decode(gaze_seq, word_boxes, cm_real, t_pom, use_ovp=True)
        _, _, surp_rec, _ = evaluate_word_and_recovery(targets, idx, line_by_word, drift[1], drift_y)

        # 3a. Baseline (Spatial Only, 2D nearest word) — keep consistent with full-corpus benchmark
        idx_nb = NearestBoundingBoxDecoder().decode(gaze_seq, word_boxes)
        _, _, base_2d_rec, _ = evaluate_word_and_recovery(targets, idx_nb, line_by_word, None, drift_y)

        # 3b. Diagnostic baseline (Y-only snap) — isolates pure "line-locking" under vertical drift
        idx_y = decode_nearest_y_only(gaze_seq, word_boxes)
        _, _, base_y_rec, _ = evaluate_word_and_recovery(targets, idx_y, line_by_word, None, drift_y)

        return {
            "Subject": sub, "Lang": lang, "Trial": trial, "Drift_Y": drift_y,
            "STOCK-T_Edge_Rec": edge_rec,
            "STOCK-T_Surprisal_Rec": surp_rec,
            # Baseline_Rec stays as the canonical 2D spatial baseline for consistency across scripts
            "Baseline_Rec": base_2d_rec,
            "BaselineY_Rec": base_y_rec,
        }
    except Exception as e:
        return None

def plot_noise_tolerance(df_results):
    print("Generating Noise Tolerance Curve...")
    
    # 計算每個漂移級別的平均還原率（略過 Subject/Lang/Trial 等非數值欄）
    df_agg = df_results.groupby("Drift_Y", as_index=False)[
        ["STOCK-T_Edge_Rec", "STOCK-T_Surprisal_Rec", "Baseline_Rec"]
    ].mean()
    
    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    plt.figure(figsize=(7, 5))
    
    plt.plot(df_agg['Drift_Y'], df_agg['STOCK-T_Edge_Rec'], marker='o', linewidth=2.5, color='#2ca25f', label='STOCK-T (Edge/Uniform)')
    plt.plot(df_agg['Drift_Y'], df_agg['STOCK-T_Surprisal_Rec'], marker='s', linewidth=2.5, color='#2b8cbe', label='STOCK-T (Surprisal)')
    plt.plot(df_agg['Drift_Y'], df_agg['Baseline_Rec'], marker='^', linewidth=2.5, color='#de2d26', linestyle='--', label='Baseline (Spatial Only)')
    
    plt.title('Noise Tolerance: OVP Washout Effect', fontweight='bold', pad=15)
    plt.xlabel('Hardware Vertical Drift (px)', fontweight='bold')
    plt.ylabel('Line recovery rate (%)', fontweight='bold')
    plt.ylim(-5, 105)
    plt.xticks(DRIFT_LEVELS)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower left')
    
    # 標示出 "Washout 臨界點"
    plt.axvline(x=30, color='gray', linestyle=':', alpha=0.5)
    plt.text(32, 85, 'Washout\nThreshold', color='gray', fontsize=10)

    # 動態創建 NeurIPS 的 figures 目錄並存檔
    neurips_fig_dir = os.path.join(PROJECT_ROOT, "docs", "NeurIPS", "figures")
    os.makedirs(neurips_fig_dir, exist_ok=True)
    plot_path = os.path.join(neurips_fig_dir, "fig_noise_degradation.pdf")
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"Plot directly saved for LaTeX inclusion at: {plot_path}")

def run_noise_tolerance_experiment_default():
    """Back-compat entrypoint (full run)."""
    run_noise_tolerance_experiment(test_mode=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run a small subset for sanity checks (no plot).")
    parser.add_argument("--test-subjects", type=int, default=2, help="Subjects per Lang in --test mode.")
    parser.add_argument("--test-trials", type=int, default=2, help="Trials per subject in --test mode.")
    args = parser.parse_args()

    if args.test:
        run_noise_tolerance_experiment(
            test_mode=True,
            limit_subjects_per_lang=args.test_subjects,
            limit_trials_per_subject=args.test_trials,
        )
    else:
        run_noise_tolerance_experiment_default()