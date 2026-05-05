import os
import sys
import pandas as pd
import numpy as np
import traceback
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns

# 確保專案根目錄正確
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.geco.core.transition_model import ReadingTransitionMatrix, PsycholinguisticTransitionMatrix
from scripts.geco.core.em_calibration import AutoCalibratingDecoder
from scripts.geco.core.baseline_decoders import NearestBoundingBoxDecoder, StandardKalmanDecoder

# 參數設定
SIGMA_FWD = 0.8
SIGMA_REG = 1.5
GAMMA = 0.3
SIGMA_X = 40.0
SIGMA_Y = 30.0

# 測試不同的垂直漂移強度 (從完美的 0px 到極端的 60px)
DRIFT_LEVELS = [0.0, 15.0, 30.0, 45.0, 60.0]

def inject_noise(df, drift_y):
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

def process_single_trial_with_drift(args):
    lang, sub, trial, layout_path, fixations_path, drift_y = args
    np.random.seed(abs(hash(sub + trial + str(drift_y))) % (2**32 - 1)) 
    
    try:
        df_layout = pd.read_csv(layout_path)
        df_fixations = pd.read_csv(fixations_path)
        
        if df_layout.empty or df_fixations.empty: return None
            
        df_fixations = df_fixations.rename(columns={'fixation_x': 'true_x', 'fixation_y': 'true_y'})
        df_fixations = inject_noise(df_fixations, drift_y)
        if df_fixations.empty: return None
        
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
        _, _, edge_rec = evaluate_metrics(targets, idx, drift[1], drift_y)

        # 2. STOCK-T_Surprisal (Real CM + POM)
        idx, drift = cal.calibrate_and_decode(gaze_seq, word_boxes, cm_real, t_pom, use_ovp=True)
        _, _, surp_rec = evaluate_metrics(targets, idx, drift[1], drift_y)

        # 3. Baseline (w/o Temp / Spatial Only)
        idx_nb = NearestBoundingBoxDecoder().decode(gaze_seq, word_boxes)
        _, _, base_rec = evaluate_metrics(targets, idx_nb, 0, drift_y)

        return {
            "Subject": sub, "Lang": lang, "Trial": trial, "Drift_Y": drift_y,
            "STOCK-T_Edge_Rec": edge_rec,
            "STOCK-T_Surprisal_Rec": surp_rec,
            "Baseline_Rec": base_rec
        }
    except Exception as e:
        return None

def plot_noise_tolerance(df_results):
    print("Generating Noise Tolerance Curve...")
    
    # 計算每個漂移級別的平均還原率
    df_agg = df_results.groupby('Drift_Y').mean().reset_index()
    
    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    plt.figure(figsize=(7, 5))
    
    plt.plot(df_agg['Drift_Y'], df_agg['STOCK-T_Edge_Rec'], marker='o', linewidth=2.5, color='#2ca25f', label='STOCK-T (Edge/Uniform)')
    plt.plot(df_agg['Drift_Y'], df_agg['STOCK-T_Surprisal_Rec'], marker='s', linewidth=2.5, color='#2b8cbe', label='STOCK-T (Surprisal)')
    plt.plot(df_agg['Drift_Y'], df_agg['Baseline_Rec'], marker='^', linewidth=2.5, color='#de2d26', linestyle='--', label='Baseline (Spatial Only)')
    
    plt.title('Noise Tolerance: OVP Washout Effect', fontweight='bold', pad=15)
    plt.xlabel('Hardware Vertical Drift (px)', fontweight='bold')
    plt.ylabel('Trajectory Recovery Rate (%)', fontweight='bold')
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

def run_noise_tolerance_experiment():
    benchmark_dir = os.path.join(PROJECT_ROOT, "data", "geco", "benchmark")
    os.makedirs(benchmark_dir, exist_ok=True)
    
    tasks = []
    # 收集任務
    for lang in ["L1", "L2"]:
        pop_dir = os.path.join(PROJECT_ROOT, f"data/geco/population/{lang}")
        if not os.path.exists(pop_dir): continue
        for sub in os.listdir(pop_dir):
            sub_dir = os.path.join(pop_dir, sub)
            if not os.path.isdir(sub_dir): continue
            for trial in [d for d in os.listdir(sub_dir) if d.startswith("trial_")]:
                layout_path = os.path.join(sub_dir, trial, "layout.csv")
                fixations_path = os.path.join(sub_dir, trial, "fixations.csv")
                if os.path.exists(layout_path) and os.path.exists(fixations_path):
                    # 對每個 Trial，展開 5 種不同的漂移強度
                    for drift in DRIFT_LEVELS:
                        tasks.append((lang, sub, trial, layout_path, fixations_path, drift))
                        
    print(f"Total experiment runs scheduled: {len(tasks)} (Trials x Drift Levels)")
    
    all_results = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_single_trial_with_drift, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Running Drift Stress Test"):
            res = future.result()
            if res: all_results.append(res)

    df_results = pd.DataFrame(all_results)
    csv_path = os.path.join(benchmark_dir, "noise_tolerance_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nExperiment finished. Data saved to {csv_path}")
    
    # 直接畫圖並存到 docs\NeurIPS\figures
    plot_noise_tolerance(df_results)

if __name__ == "__main__":
    run_noise_tolerance_experiment()