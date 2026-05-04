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
    if total == 0: return 0.0, 0.0, 0.0
    acc = sum(1 for t, p in zip(target_indices, predicted_indices) if t == p) / total * 100
    top3_acc = sum(1 for t, p in zip(target_indices, predicted_indices) if abs(int(t) - int(p)) <= 1) / total * 100
    recovery = 100.0 if abs(estimated_drift - true_drift) < 15.0 else 0.0
    return acc, top3_acc, recovery

def run_population_ablation_trial5():
    np.random.seed(42)
    # 用來存放所有成功的 Trial 紀錄 (Flattened)
    all_trial_results = []
    
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
            
            # 🛡️ 加入 try-except 防止單一受試者資料損毀導致程式崩潰
            try:
                df_layout = pd.read_csv(layout_path)
                df_fixations = pd.read_csv(fixations_path)
                
                if df_layout.empty or df_fixations.empty:
                    continue
                    
                df_fixations = df_fixations.rename(columns={'fixation_x': 'true_x', 'fixation_y': 'true_y'})
                df_fixations = inject_noise(df_fixations)
                
                # [訊號平滑] window_size=3
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
                
                # 1. Full STOCK-T
                t_pom = PsycholinguisticTransitionMatrix(sigma_fwd=SIGMA_FWD, sigma_reg=SIGMA_REG, gamma=GAMMA).build_matrix(len(df_layout), cm_real)
                idx, drift = cal.calibrate_and_decode(gaze_seq, word_boxes, cm_real, t_pom, use_ovp=True)
                full_acc, full_top3, full_rec = evaluate_metrics(targets, idx, drift[1], DRIFT_Y)

                # 2. w/o CM (Edge-Optimized)
                idx, drift = cal.calibrate_and_decode(gaze_seq, word_boxes, cm_uniform, t_pom, use_ovp=True)
                no_cm_acc, no_cm_top3, no_cm_rec = evaluate_metrics(targets, idx, drift[1], DRIFT_Y)

                # 3. w/o POM
                t_rule = ReadingTransitionMatrix().build_matrix(cm_real, is_L2_reader=(lang=="L2"))
                idx, drift = cal.calibrate_and_decode(gaze_seq, word_boxes, cm_real, t_rule, use_ovp=True)
                no_pom_acc, no_pom_top3, no_pom_rec = evaluate_metrics(targets, idx, drift[1], DRIFT_Y)

                # 4. w/o EM
                idx_k = StandardKalmanDecoder().decode(gaze_seq, word_boxes)
                no_em_acc, no_em_top3, _ = evaluate_metrics(targets, idx_k, 0, DRIFT_Y)

                # 5. w/o Temp
                idx_nb = NearestBoundingBoxDecoder().decode(gaze_seq, word_boxes)
                no_temp_acc, no_temp_top3, _ = evaluate_metrics(targets, idx_nb, 0, DRIFT_Y)

                # 將此回合結果存入列表
                all_trial_results.append({
                    "Subject": sub, "Lang": lang,
                    "Full_Acc": full_acc, "Full_Top3": full_top3, "Full_Rec": full_rec,
                    "NoCM_Acc": no_cm_acc, "NoCM_Top3": no_cm_top3, "NoCM_Rec": no_cm_rec,
                    "NoPOM_Acc": no_pom_acc, "NoPOM_Top3": no_pom_top3, "NoPOM_Rec": no_pom_rec,
                    "NoEM_Acc": no_em_acc, "NoEM_Top3": no_em_top3, "NoEM_Rec": 0.0,
                    "NoTemp_Acc": no_temp_acc, "NoTemp_Top3": no_temp_top3, "NoTemp_Rec": 0.0
                })
                
                # 🛡️ 實時存檔 (Checkpointing)：每跑完一個人就存一次，當機也不怕
                pd.DataFrame(all_trial_results).to_csv("interim_results.csv", index=False)

            except Exception as e:
                print(f"\n⚠️ Error processing {sub}: {str(e)}")
                with open("error_log.txt", "a") as f:
                    f.write(f"Error in {sub}: {traceback.format_exc()}\n")
                continue

    # 計算最終全局平均
    df_results = pd.DataFrame(all_trial_results)
    
    print("\n" + "="*80)
    print("FINAL NEURIPS TABLE RESULTS (Averaged across valid trials)")
    print("="*80)
    
    for variant, prefix in zip(["STOCK-T (Surprisal-Guided)", "STOCK-T (Edge-Optimized)", "w/o POM", "w/o EM", "w/o Temp"], 
                               ["Full", "NoCM", "NoPOM", "NoEM", "NoTemp"]):
        acc = df_results[f"{prefix}_Acc"].mean()
        top3 = df_results[f"{prefix}_Top3"].mean()
        rec = df_results[f"{prefix}_Rec"].mean()
        print(f"{variant:<30} | Acc: {acc:>5.2f}% | Top-3: {top3:>5.2f}% | Recovery: {rec:>5.2f}%")

if __name__ == "__main__":
    run_population_ablation_trial5()