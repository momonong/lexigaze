import os
import sys
import pandas as pd
import numpy as np
import traceback
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.geco.core.transition_model import ReadingTransitionMatrix, PsycholinguisticTransitionMatrix
from scripts.geco.core.viterbi_decoder import viterbi_gaze_decode
from scripts.geco.core.em_calibration import AutoCalibratingDecoder
from scripts.geco.core.baseline_decoders import NearestBoundingBoxDecoder, StandardKalmanDecoder

# Params
SIGMA_FWD = 0.8
SIGMA_REG = 1.5
GAMMA = 0.3
DRIFT_Y = 45.0
SIGMA_X = 40.0
SIGMA_Y = 30.0

def inject_noise(df, drift_y=DRIFT_Y):
    df['true_x'] = pd.to_numeric(df['true_x'], errors='coerce')
    df['true_y'] = pd.to_numeric(df['true_y'], errors='coerce')
    df = df.dropna(subset=['true_x', 'true_y']).copy()
    n = len(df)
    df['noisy_x'] = df['true_x'] + np.random.normal(0, SIGMA_X, n)
    df['noisy_y'] = df['true_y'] + np.random.normal(0, SIGMA_Y, n) + drift_y
    return df

def evaluate_metrics(target_indices, predicted_indices, estimated_drift, true_drift):
    total = len(target_indices)
    if total == 0: return 0.0, 0.0, 0.0
    acc = sum(1 for t, p in zip(target_indices, predicted_indices) if t == p) / total * 100
    top3_acc = sum(1 for t, p in zip(target_indices, predicted_indices) if abs(int(t) - int(p)) <= 1) / total * 100
    recovery = 100.0 if abs(estimated_drift - true_drift) < 15.0 else 0.0
    return acc, top3_acc, recovery

def run_full_corpus():
    np.random.seed(42)
    all_trial_results = []
    
    benchmark_dir = "data/geco/benchmark"
    os.makedirs(benchmark_dir, exist_ok=True)
    
    for lang in ["L1", "L2"]:
        pop_dir = f"data/geco/population/{lang}"
        if not os.path.exists(pop_dir):
            continue
            
        subjects = sorted(os.listdir(pop_dir))
        for sub in tqdm(subjects, desc=f"Evaluating {lang}"):
            sub_dir = os.path.join(pop_dir, sub)
            if not os.path.isdir(sub_dir): continue
            
            trials = [d for d in os.listdir(sub_dir) if d.startswith("trial_")]
            for trial in trials:
                trial_dir = os.path.join(sub_dir, trial)
                layout_path = os.path.join(trial_dir, "layout.csv")
                fixations_path = os.path.join(trial_dir, "fixations.csv")
                
                if not os.path.exists(layout_path) or not os.path.exists(fixations_path):
                    continue
                
                try:
                    df_layout = pd.read_csv(layout_path)
                    df_fixations = pd.read_csv(fixations_path)
                    
                    if df_layout.empty or df_fixations.empty:
                        continue
                        
                    df_fixations = df_fixations.rename(columns={'fixation_x': 'true_x', 'fixation_y': 'true_y'})
                    
                    # Academic Reproducibility: Use the same hash-based seed logic as noise_tolerance script
                    np.random.seed(abs(hash(sub + trial + str(DRIFT_Y))) % (2**32 - 1))
                    
                    df_fixations = inject_noise(df_fixations, DRIFT_Y)
                    if df_fixations.empty:
                        continue
                    
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
                    
                    # 1. STOCK-T_Edge
                    t_pom = PsycholinguisticTransitionMatrix(sigma_fwd=SIGMA_FWD, sigma_reg=SIGMA_REG, gamma=GAMMA).build_matrix(len(df_layout), cm_real)
                    idx, drift = cal.calibrate_and_decode(gaze_seq, word_boxes, cm_uniform, t_pom, use_ovp=True)
                    edge_acc, edge_top3, edge_rec = evaluate_metrics(targets, idx, drift[1], DRIFT_Y)

                    # 2. STOCK-T_Surprisal
                    idx, drift = cal.calibrate_and_decode(gaze_seq, word_boxes, cm_real, t_pom, use_ovp=True)
                    surp_acc, surp_top3, surp_rec = evaluate_metrics(targets, idx, drift[1], DRIFT_Y)

                    # 3. w/o_POM
                    t_rule = ReadingTransitionMatrix().build_matrix(cm_real, is_L2_reader=(lang=="L2"))
                    idx, drift = cal.calibrate_and_decode(gaze_seq, word_boxes, cm_uniform, t_rule, use_ovp=True)
                    no_pom_acc, no_pom_top3, no_pom_rec = evaluate_metrics(targets, idx, drift[1], DRIFT_Y)

                    # 4. w/o_EM
                    idx_k = StandardKalmanDecoder().decode(gaze_seq, word_boxes)
                    no_em_acc, no_em_top3, _ = evaluate_metrics(targets, idx_k, 0, DRIFT_Y)

                    # 5. w/o_Temp
                    idx_nb = NearestBoundingBoxDecoder().decode(gaze_seq, word_boxes)
                    no_temp_acc, no_temp_top3, _ = evaluate_metrics(targets, idx_nb, 0, DRIFT_Y)

                    all_trial_results.append({
                        "Subject": sub, "Lang": lang, "Trial": trial,
                        "STOCK-T_Edge_Acc": edge_acc, "STOCK-T_Edge_Top3": edge_top3, "STOCK-T_Edge_Rec": edge_rec,
                        "STOCK-T_Surprisal_Acc": surp_acc, "STOCK-T_Surprisal_Top3": surp_top3, "STOCK-T_Surprisal_Rec": surp_rec,
                        "w/o_POM_Acc": no_pom_acc, "w/o_POM_Top3": no_pom_top3, "w/o_POM_Rec": no_pom_rec,
                        "w/o_EM_Acc": no_em_acc, "w/o_EM_Top3": no_em_top3, "w/o_EM_Rec": 0.0,
                        "w/o_Temp_Acc": no_temp_acc, "w/o_Temp_Top3": no_temp_top3, "w/o_Temp_Rec": 0.0
                    })
                    
                except Exception as e:
                    with open("error.log", "a") as f:
                        f.write(f"Error in {sub}/{trial}: {traceback.format_exc()}\n")
                    continue

    df_results = pd.DataFrame(all_trial_results)
    df_results.to_csv("data/geco/benchmark/full_corpus_results.csv", index=False)
    print("Evaluation completed. Saved to data/geco/benchmark/full_corpus_results.csv")

if __name__ == "__main__":
    run_full_corpus()