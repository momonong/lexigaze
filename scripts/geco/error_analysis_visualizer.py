import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.geco.core.transition_model import PsycholinguisticTransitionMatrix
from scripts.geco.core.viterbi_decoder import viterbi_gaze_decode
from scripts.geco.core.em_calibration import AutoCalibratingDecoder
from scripts.geco.core.dynamic_field import DynamicCognitiveField

# 1. Configuration
DATA_PATH = "data/geco/geco_pp01_cognitive_mass.csv"
OUTPUT_PLOT_PATH = "docs/error_plots/trial5_analysis.png"

# Optimal Params (Skill 11)
SIGMA_FWD = 0.8
SIGMA_REG = 1.5
GAMMA = 0.3

# Noise Model
DRIFT_Y = 45.0
SIGMA_X = 40.0
SIGMA_Y = 30.0

def inject_noise(df):
    """Inject systematic drift and Gaussian jitter."""
    np.random.seed(42)
    df['noisy_x'] = df['true_x'] + np.random.normal(0, SIGMA_X, len(df))
    df['noisy_y'] = df['true_y'] + np.random.normal(0, SIGMA_Y, len(df)) + DRIFT_Y
    return df

def run_error_analysis():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    df = inject_noise(df)
    
    # Run on ALL words to ensure consistency with benchmark
    word_boxes = [[row['true_x']-20, row['true_y']-15, row['true_x']+20, row['true_y']+15] for _, row in df.iterrows()]
    base_cm = df['cognitive_mass'].values
    gaze_sequence = df[['noisy_x', 'noisy_y']].values
    true_words = df['WORD'].values
    target_indices = np.arange(len(df))
    
    # Run Config D (Optimized POM + EM + OVP)
    pom_builder = PsycholinguisticTransitionMatrix(sigma_fwd=SIGMA_FWD, sigma_reg=SIGMA_REG, gamma=GAMMA)
    t_matrix = pom_builder.build_matrix(len(df), base_cm)
    
    calibrator = AutoCalibratingDecoder(calibration_window_size=30)
    predicted_indices, drift = calibrator.calibrate_and_decode(
        gaze_sequence, word_boxes, base_cm, t_matrix, 
        sigma_gaze=[SIGMA_X, SIGMA_Y], use_ovp=True
    )
    
    # Calculate drift-corrected gaze for plotting
    corrected_gaze = gaze_sequence - np.array(drift)
    
    # --- Visualization Subset (First 70 fixations) ---
    VIEW_LIMIT = 70
    df_subset = df.iloc[:VIEW_LIMIT].copy()
    sub_predicted = predicted_indices[:VIEW_LIMIT]
    sub_gaze = gaze_sequence[:VIEW_LIMIT]
    sub_corrected = corrected_gaze[:VIEW_LIMIT]
    sub_true_indices = target_indices[:VIEW_LIMIT]
    sub_word_boxes = word_boxes[:VIEW_LIMIT]
    sub_true_words = true_words[:VIEW_LIMIT]
    
    # Failure Logging
    print("\n" + "="*50)
    print(f"❌ FAILURE LOG (First {VIEW_LIMIT} words)")
    print("="*50)
    for t in range(VIEW_LIMIT):
        true_idx = sub_true_indices[t]
        pred_idx = sub_predicted[t]
        
        if true_idx != pred_idx:
            note = "Relaxed Match" if abs(true_idx - pred_idx) <= 1 else "COMPLETE MISS"
            print(f"Time {t:2d} | True: \"{sub_true_words[t]:10s}\" (idx:{true_idx:2d}) | Pred: \"{true_words[pred_idx]:10s}\" (idx:{pred_idx:2d}) | {note}")
    
    # --- Visualization ---
    plt.figure(figsize=(20, 10))
    ax = plt.gca()
    
    # 1. Plot Word Bounding Boxes (of ALL words that are in view)
    # Filter word boxes that are within the min/max X/Y of the first 70 words
    min_x, max_x = df_subset['true_x'].min() - 100, df_subset['true_x'].max() + 100
    min_y, max_y = df_subset['true_y'].min() - 100, df_subset['true_y'].max() + 100
    
    dfield = DynamicCognitiveField(word_boxes, base_cm, use_ovp=True)
    ovp_centers = dfield.word_centers
    
    for i, box in enumerate(word_boxes):
        if box[0] < min_x or box[2] > max_x or box[1] < min_y or box[3] > max_y:
            continue
            
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                                 linewidth=1, edgecolor='gray', facecolor='none', alpha=0.3)
        ax.add_patch(rect)
        # Text
        plt.text(box[0], box[1]-5, f"{i}:{df.iloc[i]['WORD']}", fontsize=8, color='gray')
        # OVP Center
        plt.plot(ovp_centers[i, 0], ovp_centers[i, 1], 'rx', markersize=3, alpha=0.5)

    # 2. Plot Raw Gaze (Faint Red dots)
    plt.scatter(sub_gaze[:, 0], sub_gaze[:, 1], c='red', s=10, alpha=0.2, label='Raw Gaze (+45px Drift)')
    
    # 3. Plot Drift-Corrected Gaze (Orange dots)
    plt.scatter(sub_corrected[:, 0], sub_corrected[:, 1], c='orange', s=15, alpha=0.4, label='Drift-Corrected Gaze (EM)')

    # 4. Plot Ground Truth Path (Solid Green Line)
    # Centers of words user actually looked at
    gt_centers = np.array([[ (word_boxes[idx][0] + word_boxes[idx][2]) / 2, (word_boxes[idx][1] + word_boxes[idx][3]) / 2 ] for idx in sub_true_indices])
    plt.plot(gt_centers[:, 0], gt_centers[:, 1], 'g-', linewidth=2, alpha=0.6, label='Ground Truth Path')
    
    # 5. Plot Predicted Path (Dashed Blue Line)
    pred_centers = ovp_centers[sub_predicted]
    plt.plot(pred_centers[:, 0], pred_centers[:, 1], 'b--', linewidth=2, alpha=0.8, label='STOCK-T Predicted Path (Config D)')

    plt.title(f"LexiGaze Error Analysis: pp01 Trial 5 (Config D, Optimized POM)", fontsize=16)
    plt.xlabel("Screen X (px)")
    plt.ylabel("Screen Y (px)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.gca().invert_yaxis()
    
    os.makedirs(os.path.dirname(OUTPUT_PLOT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
    print(f"\n✅ Visualization saved to {OUTPUT_PLOT_PATH}")
    plt.close()

if __name__ == "__main__":
    run_error_analysis()
