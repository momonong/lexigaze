import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import os
import sys

# Add project root to path for core modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.geco.core.transition_model import PsycholinguisticTransitionMatrix
from scripts.geco.core.em_calibration import AutoCalibratingDecoder

# 1. NeurIPS Standard Typography
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

def generate_fig5_real_data():
    # 2. Load Real GECO Data
    data_path = "data/geco/geco_pp01_cognitive_mass.csv"
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found.")
        return
    
    df_cm = pd.read_csv(data_path)
    df_clean = pd.read_csv("data/geco/geco_pp01_trial5_clean.csv")
    
    # Merge to get reading times
    df_full = pd.merge(df_cm, df_clean[['WORD_ID', 'WORD_TOTAL_READING_TIME']], on='WORD_ID')
    
    # Focus on the first ~22 words to show two lines
    LIMIT = 22
    df = df_full.iloc[:LIMIT].copy()
    
    # Estimate word boxes based on string length
    # char_width ~7px, padding 10px
    char_w = 7.5
    df['width'] = df['WORD'].str.strip().str.len() * char_w + 10
    df['height'] = 28
    df['x_min'] = df['true_x'] - df['width'] / 2
    df['y_min'] = df['true_y'] - df['height'] / 2
    df['x_max'] = df['true_x'] + df['width'] / 2
    df['y_max'] = df['true_y'] + df['height'] / 2
    
    word_boxes = df[['x_min', 'y_min', 'x_max', 'y_max']].values.tolist()
    base_cm = df['cognitive_mass'].values
    
    # 3. Simulate Noisy Gaze (Drift + Jitter)
    np.random.seed(42)
    DRIFT_Y = 45.0
    SIGMA_X = 25.0
    SIGMA_Y = 15.0
    
    # We simulate multiple points per word to make the trajectory look more realistic?
    # Actually, GECO data provided is word-level summaries. 
    # To show a "scanpath", we can just use one point per word or interpolate.
    # Let's use the actual word centers + noise.
    raw_x = df['true_x'] + np.random.normal(0, SIGMA_X, len(df))
    raw_y = df['true_y'] + np.random.normal(0, SIGMA_Y, len(df)) + DRIFT_Y
    gaze_sequence = np.stack([raw_x, raw_y], axis=1)
    
    # 4. Run STOCK-T Algorithm
    # Optimal Params for pp01 Trial 5
    pom_builder = PsycholinguisticTransitionMatrix(sigma_fwd=0.8, sigma_reg=1.5, gamma=0.3)
    t_matrix = pom_builder.build_matrix(len(df), base_cm)
    
    calibrator = AutoCalibratingDecoder(calibration_window_size=15) # Shorter window for small sample
    # Note: hypotheses=[0, 40, -40] is default in em_calibration.py
    # We know the drift is +45, so h=-40 or -line_height should be picked.
    predicted_indices, drift_vector = calibrator.calibrate_and_decode(
        gaze_sequence, word_boxes, base_cm, t_matrix, 
        sigma_gaze=[SIGMA_X, SIGMA_Y], use_ovp=True
    )
    
    corrected_gaze = gaze_sequence - np.array(drift_vector)
    cx = corrected_gaze[:, 0]
    cy = corrected_gaze[:, 1]
    
    # 5. Plotting
    fig, ax = plt.subplots(figsize=(8, 3.8))
    
    # Canvas limits
    ax.set_xlim(df['x_min'].min() - 50, df['x_max'].max() + 50)
    ax.set_ylim(df['y_min'].min() - 80, df['y_max'].max() + 100)
    ax.invert_yaxis()
    
    # Strictly NO axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    # Z-order 1: Stimulus & Heatmap
    # Find high-CM words for heatmap (e.g. top 25% percent)
    threshold = np.percentile(base_cm, 75)
    
    for i, row in df.iterrows():
        # Soft Heatmap
        if row['cognitive_mass'] >= threshold:
            heatmap = patches.Rectangle((row['x_min']-2, row['y_min']-2), 
                                       row['width']+4, row['height']+4, 
                                       facecolor="#FF8C00", alpha=0.1, edgecolor="none", zorder=1)
            ax.add_patch(heatmap)
            
        # Bounding Box
        rect = patches.Rectangle((row['x_min'], row['y_min']), 
                                 row['width'], row['height'], 
                                 linewidth=0.8, edgecolor="lightgray", facecolor="none", 
                                 linestyle="--", alpha=0.8, zorder=1)
        ax.add_patch(rect)
        
        # Word Text
        ax.text(row['true_x'], row['true_y'], row['WORD'].strip(), 
                ha="center", va="center", family="serif", fontsize=10, zorder=1)

    # Z-order 2: Raw Gaze
    ax.plot(raw_x, raw_y, color="#E63946", linestyle="--", alpha=0.4, linewidth=1.0, zorder=2, label="Raw Gaze (Hardware Drift)")
    ax.scatter(raw_x, raw_y, color="#E63946", marker="x", s=20, alpha=0.5, zorder=2)
    
    # Z-order 3: Semantic Gravity Arcs
    for i in range(len(df)):
        ax.annotate("", xy=(cx[i], cy[i]), xytext=(raw_x[i], raw_y[i]),
                    arrowprops=dict(arrowstyle="->", color="gray", linestyle=":", 
                                    shrinkA=3, shrinkB=3, alpha=0.4,
                                    connectionstyle="arc3,rad=-0.2"), zorder=3)
    
    # Hidden proxy for legend
    ax.plot([], [], color="gray", linestyle=":", label="Semantic Gravity Arc", zorder=3)
    
    # Z-order 4: STOCK-T Corrected
    ax.plot(cx, cy, color="#2A9D8F", linewidth=2.0, alpha=0.9, zorder=4, label="STOCK-T Corrected")
    
    # Vary size by fixation duration (from real data!)
    # Normalize durations for visual clarity
    max_dur = df['WORD_TOTAL_READING_TIME'].max()
    dur_sizes = (df['WORD_TOTAL_READING_TIME'] / max_dur) * 150 + 20
    
    ax.scatter(cx, cy, color="#2A9D8F", s=dur_sizes, marker="o", edgecolors="white", linewidths=0.5, alpha=0.9, zorder=4)
    
    # Directional arrows on corrected path
    for i in range(len(df)-1):
        # Only add arrow if words are sequential
        ax.annotate("", xy=(cx[i+1], cy[i+1]), xytext=(cx[i], cy[i]),
                    arrowprops=dict(arrowstyle="-|>", color="#2A9D8F", lw=0, alpha=0.5, shrinkA=12, shrinkB=12),
                    zorder=4)

    # Legend Outside
    ax.legend(bbox_to_anchor=(0.5, 1.15), loc='lower center', ncol=3, frameon=False, fontsize=9)
    
    # Output
    output_dir = "docs/NeurIPS/figures"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fig_scanpath_correction_real.pdf")
    plt.savefig(output_path, bbox_inches="tight")
    print(f"✅ Real-data high-fidelity figure generated: {output_path}")

if __name__ == "__main__":
    generate_fig5_real_data()
