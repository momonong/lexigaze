import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.geco.core.transition_model import ReadingTransitionMatrix, PsycholinguisticTransitionMatrix
from scripts.geco.core.viterbi_decoder import viterbi_gaze_decode
from scripts.geco.core.em_calibration import AutoCalibratingDecoder
from scripts.geco.core.baseline_decoders import NearestBoundingBoxDecoder, StandardKalmanDecoder

# 1. Configuration
POP_DIR = "data/geco/population"
LANGUAGES = ["L1", "L2"]

# Optimal Params (Skill 11/14)
SIGMA_FWD = 0.8
SIGMA_REG = 1.5
GAMMA = 0.3

# Noise Model (NeurIPS Stress Test)
DRIFT_Y = 45.0
SIGMA_X = 40.0
SIGMA_Y = 30.0

def inject_noise(df):
    np.random.seed(42)
    df['noisy_x'] = df['true_x'] + np.random.normal(0, SIGMA_X, len(df))
    df['noisy_y'] = df['true_y'] + np.random.normal(0, SIGMA_Y, len(df)) + DRIFT_Y
    return df

def evaluate_metrics(target_indices, predicted_indices, estimated_drift, true_drift):
    total = len(target_indices)
    acc = sum(1 for t, p in zip(target_indices, predicted_indices) if t == p) / total * 100
    # Recovery rate: is estimated drift within 15px of true drift?
    recovery = 100.0 if abs(estimated_drift - true_drift) < 15.0 else 0.0
    return acc, recovery

def run_population_ablation():
    # results[variant][lang] -> list of accuracies
    variants = ["Full", "w/o CM", "w/o POM", "w/o EM", "w/o Temp"]
    results = {v: {l: [] for l in LANGUAGES + ["Rec"]} for v in variants}

    feature_files = glob.glob(f"{POP_DIR}/*/*/*/features.csv")
    print(f"🔍 Found {len(feature_files)} trial files across all subjects.")

    pbar = tqdm(total=len(feature_files), desc="Evaluating Population")

    for path in feature_files:
        path_parts = path.replace("\\", "/").split("/")
        # Extract lang from path parts
        lang = "L1" if "L1" in path_parts else "L2"
        
        df = pd.read_csv(path)
        df = inject_noise(df)
        
        cm_real = df['cognitive_mass'].values
        cm_uniform = np.ones(len(df)) * 2.5
        
        word_boxes = [[row['true_x']-20, row['true_y']-15, row['true_x']+20, row['true_y']+15] for _, row in df.iterrows()]
        gaze_seq = df[['noisy_x', 'noisy_y']].values
        targets = np.arange(len(df))

        # 1. Full STOCK-T
        t_pom = PsycholinguisticTransitionMatrix(sigma_fwd=SIGMA_FWD, sigma_reg=SIGMA_REG, gamma=GAMMA).build_matrix(len(df), cm_real)
        cal = AutoCalibratingDecoder()
        idx, drift = cal.calibrate_and_decode(gaze_seq, word_boxes, cm_real, t_pom, use_ovp=True)
        acc, rec = evaluate_metrics(targets, idx, drift[1], DRIFT_Y)
        results["Full"][lang].append(acc); results["Full"]["Rec"].append(rec)

        # 2. w/o CM
        idx, drift = cal.calibrate_and_decode(gaze_seq, word_boxes, cm_uniform, t_pom, use_ovp=True)
        acc, rec = evaluate_metrics(targets, idx, drift[1], DRIFT_Y)
        results["w/o CM"][lang].append(acc); results["w/o CM"]["Rec"].append(rec)

        # 3. w/o POM (Use rule-based)
        t_rule = ReadingTransitionMatrix().build_matrix(cm_real, is_L2_reader=(lang=="L2"))
        idx, drift = cal.calibrate_and_decode(gaze_seq, word_boxes, cm_real, t_rule, use_ovp=True)
        acc, rec = evaluate_metrics(targets, idx, drift[1], DRIFT_Y)
        results["w/o POM"][lang].append(acc); results["w/o POM"]["Rec"].append(rec)

        # 4. w/o EM (Standard Kalman)
        idx_k = StandardKalmanDecoder().decode(gaze_seq, word_boxes)
        acc, _ = evaluate_metrics(targets, idx_k, 0, DRIFT_Y)
        results["w/o EM"][lang].append(acc); results["w/o EM"]["Rec"].append(0.0)

        # 5. w/o Temp (Nearest Box)
        idx_nb = NearestBoundingBoxDecoder().decode(gaze_seq, word_boxes)
        acc, _ = evaluate_metrics(targets, idx_nb, 0, DRIFT_Y)
        results["w/o Temp"][lang].append(acc); results["w/o Temp"]["Rec"].append(0.0)
        
        pbar.update(1)
    
    pbar.close()

    # Summary Table
    print("\n" + "="*70)
    print("FINAL POPULATION ABLATION TABLE (ALL Subjects)")
    print("="*70)
    print(f"{'Model Variant':<25} | {'L1 Acc':<10} | {'L2 Acc':<10} | {'Rec Rate':<10}")
    print("-" * 70)
    
    summary_data = []
    for v in variants:
        l1_m = np.mean(results[v]["L1"]) if results[v]["L1"] else 0
        l2_m = np.mean(results[v]["L2"]) if results[v]["L2"] else 0
        rec_m = np.mean(results[v]["Rec"]) if results[v]["Rec"] else 0
        print(f"{v:<25} | {l1_m:>8.2f}% | {l2_m:>8.2f}% | {rec_m:>8.2f}%")
        summary_data.append([v, l1_m, l2_m, rec_m])

    return summary_data

if __name__ == "__main__":
    run_population_ablation()
