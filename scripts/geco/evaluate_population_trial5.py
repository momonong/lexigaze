import os
import sys
import pandas as pd
import numpy as np
import torch
import math
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.geco.core.transition_model import ReadingTransitionMatrix, PsycholinguisticTransitionMatrix
from scripts.geco.core.viterbi_decoder import viterbi_gaze_decode
from scripts.geco.core.em_calibration import AutoCalibratingDecoder
from scripts.geco.core.baseline_decoders import NearestBoundingBoxDecoder, StandardKalmanDecoder

# Optimal Params
SIGMA_FWD = 0.8
SIGMA_REG = 1.5
GAMMA = 0.3
DRIFT_Y = 45.0
SIGMA_X = 40.0
SIGMA_Y = 30.0

def inject_noise(df):
    np.random.seed(42)
    df['true_x'] = pd.to_numeric(df['true_x'], errors='coerce')
    df['true_y'] = pd.to_numeric(df['true_y'], errors='coerce')
    df = df.dropna(subset=['true_x', 'true_y'])
    n = len(df)
    df['noisy_x'] = df['true_x'] + np.random.normal(0, SIGMA_X, n)
    df['noisy_y'] = df['true_y'] + np.random.normal(0, SIGMA_Y, n) + DRIFT_Y
    return df

def evaluate_metrics(target_indices, predicted_indices, estimated_drift, true_drift):
    total = len(target_indices)
    if total == 0: return 0.0, 0.0
    acc = sum(1 for t, p in zip(target_indices, predicted_indices) if t == p) / total * 100
    recovery = 100.0 if abs(estimated_drift - true_drift) < 15.0 else 0.0
    return acc, recovery

def run_population_ablation_trial5():
    results = {
        "Full": {"L1": [], "L2": [], "Rec": []},
        "w/o CM": {"L1": [], "L2": [], "Rec": []},
        "w/o POM": {"L1": [], "L2": [], "Rec": []},
        "w/o EM": {"L1": [], "L2": [], "Rec": []},
        "w/o Temp": {"L1": [], "L2": [], "Rec": []}
    }

    for lang in ["L1", "L2"]:
        pop_dir = f"data/geco/population/{lang}"
        if not os.path.exists(pop_dir):
            print(f"⚠️ {lang} population directory not found.")
            continue
            
        subjects = os.listdir(pop_dir)
        for sub in tqdm(subjects, desc=f"Evaluating {lang}"):
            trial_dir = f"{pop_dir}/{sub}/trial_5"
            layout_path = f"{trial_dir}/layout.csv"
            fixations_path = f"{trial_dir}/fixations.csv"
            
            if not os.path.exists(layout_path) or not os.path.exists(fixations_path):
                continue
                
            df_layout = pd.read_csv(layout_path)
            df_fixations = pd.read_csv(fixations_path)
            df_fixations = df_fixations.rename(columns={'fixation_x': 'true_x', 'fixation_y': 'true_y'})
            df_fixations = inject_noise(df_fixations)
            
            cm_real = df_layout['cognitive_mass'].values
            cm_uniform = np.ones(len(df_layout)) * 2.5
            
            word_boxes = []
            for _, row in df_layout.iterrows():
                word_str = str(row['WORD']).strip()
                # Estimate width: ~12px per character, min 40px
                w = max(40.0, len(word_str) * 12.0)
                h = 40.0 # Standard line height estimate
                word_boxes.append([
                    row['true_x'] - w/2, 
                    row['true_y'] - h/2, 
                    row['true_x'] + w/2, 
                    row['true_y'] + h/2
                ])
            gaze_seq = df_fixations[['noisy_x', 'noisy_y']].values
            targets = df_fixations['layout_index'].values.astype(int)

            # 1. Full STOCK-T
            t_pom = PsycholinguisticTransitionMatrix(sigma_fwd=SIGMA_FWD, sigma_reg=SIGMA_REG, gamma=GAMMA).build_matrix(len(df_layout), cm_real)
            cal = AutoCalibratingDecoder()
            idx, drift = cal.calibrate_and_decode(gaze_seq, word_boxes, cm_real, t_pom, use_ovp=True)
            acc, rec = evaluate_metrics(targets, idx, drift[1], DRIFT_Y)
            results["Full"][lang].append(acc); results["Full"]["Rec"].append(rec)

            # 2. w/o CM
            idx, drift = cal.calibrate_and_decode(gaze_seq, word_boxes, cm_uniform, t_pom, use_ovp=True)
            acc, rec = evaluate_metrics(targets, idx, drift[1], DRIFT_Y)
            results["w/o CM"][lang].append(acc); results["w/o CM"]["Rec"].append(rec)

            # 3. w/o POM
            t_rule = ReadingTransitionMatrix().build_matrix(cm_real, is_L2_reader=(lang=="L2"))
            idx, drift = cal.calibrate_and_decode(gaze_seq, word_boxes, cm_real, t_rule, use_ovp=True)
            acc, rec = evaluate_metrics(targets, idx, drift[1], DRIFT_Y)
            results["w/o POM"][lang].append(acc); results["w/o POM"]["Rec"].append(rec)

            # 4. w/o EM
            idx_k = StandardKalmanDecoder().decode(gaze_seq, word_boxes)
            acc, _ = evaluate_metrics(targets, idx_k, 0, DRIFT_Y)
            results["w/o EM"][lang].append(acc); results["w/o EM"]["Rec"].append(0.0)

            # 5. w/o Temp
            idx_nb = NearestBoundingBoxDecoder().decode(gaze_seq, word_boxes)
            acc, _ = evaluate_metrics(targets, idx_nb, 0, DRIFT_Y)
            results["w/o Temp"][lang].append(acc); results["w/o Temp"]["Rec"].append(0.0)

    # Summary
    print("\n" + "="*60)
    print("POPULATION TRIAL 5 ABLATION SUMMARY")
    print("="*60)
    print(f"{'Model Variant':<25} | {'L1 Acc':<10} | {'L2 Acc':<10} | {'Rec Rate':<10}")
    print("-" * 60)
    
    for variant in ["Full", "w/o CM", "w/o POM", "w/o EM", "w/o Temp"]:
        l1_vals = results[variant]["L1"]
        l2_vals = results[variant]["L2"]
        rec_vals = results[variant]["Rec"]
        
        l1_m = np.mean(l1_vals) if l1_vals else 0.0
        l2_m = np.mean(l2_vals) if l2_vals else 0.0
        rec_m = np.mean(rec_vals) if rec_vals else 0.0
        
        print(f"{variant:<25} | {l1_m:>8.2f}% | {l2_m:>8.2f}% | {rec_m:>8.2f}%")

if __name__ == "__main__":
    run_population_ablation_trial5()
