import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path for core modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.geco.core.transition_model import PsycholinguisticTransitionMatrix
from scripts.geco.core.em_calibration import AutoCalibratingDecoder
from scripts.geco.core.dynamic_field import DynamicCognitiveField

# 1. NeurIPS Typography & Sizing
os.makedirs("docs/NeurIPS/figures", exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# 2. Configuration & Data Loading
DATA_PATH = "data/geco/geco_pp01_cognitive_mass.csv"
OUTPUT_PDF_PATH = "docs/NeurIPS/figures/trial5_analysis.pdf"

# STOCK-T Params (Skill 11 Optimal)
SIGMA_FWD = 0.8
SIGMA_REG = 1.5
GAMMA = 0.3
DRIFT_Y = 45.0
SIGMA_X = 40.0
SIGMA_Y = 30.0

def generate_trajectory_data():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found.")
        return None

    df = pd.read_csv(DATA_PATH)
    
    # Inject Noise (Line-Locking Failure Mode)
    np.random.seed(42)
    df['raw_x'] = df['true_x'] + np.random.normal(0, SIGMA_X, len(df))
    df['raw_y'] = df['true_y'] + np.random.normal(0, SIGMA_Y, len(df)) + DRIFT_Y
    
    # Define Word Boxes
    word_boxes = [[row['true_x']-20, row['true_y']-15, row['true_x']+20, row['true_y']+15] for _, row in df.iterrows()]
    base_cm = df['cognitive_mass'].values
    gaze_sequence = df[['raw_x', 'raw_y']].values
    
    # Run STOCK-T (POM + EM + OVP)
    pom_builder = PsycholinguisticTransitionMatrix(sigma_fwd=SIGMA_FWD, sigma_reg=SIGMA_REG, gamma=GAMMA)
    t_matrix = pom_builder.build_matrix(len(df), base_cm)
    
    calibrator = AutoCalibratingDecoder(calibration_window_size=30)
    # Multi-Hypothesis EM is internal to calibrator now (Skill 14)
    predicted_indices, drift = calibrator.calibrate_and_decode(
        gaze_sequence, word_boxes, base_cm, t_matrix, 
        sigma_gaze=[SIGMA_X, SIGMA_Y], use_ovp=True
    )
    
    # Get Corrected Gaze
    corrected_gaze = gaze_sequence - np.array(drift)
    df['corrected_x'] = corrected_gaze[:, 0]
    df['corrected_y'] = corrected_gaze[:, 1]
    
    # Add box coords for plotting
    df['x_min'] = df['true_x'] - 20
    df['y_min'] = df['true_y'] - 15
    df['x_max'] = df['true_x'] + 20
    df['y_max'] = df['true_y'] + 15
    
    return df, word_boxes

def render_trajectory():
    data = generate_trajectory_data()
    if data is None: return
    df, word_boxes = data
    
    # We focus on a specific segment (e.g. words 0 to 15) to show "Line-Locking" and its correction
    # The first sentence is about 11 words.
    SEGMENT_LIMIT = 15
    df_seg = df.iloc[:SEGMENT_LIMIT].copy()
    
    fig, ax = plt.subplots(figsize=(5.5, 2.0))
    
    # 1. Bounding Boxes
    # Show more boxes in the vicinity to illustrate the "Line-Locking" trap (Line 2 boxes)
    # Line 2 starts around word 11
    PLOT_LIMIT = 25 
    for i in range(min(PLOT_LIMIT, len(df))):
        row = df.iloc[i]
        rect = patches.Rectangle((row['x_min'], row['y_min']), 
                                 row['x_max'] - row['x_min'], 
                                 row['y_max'] - row['y_min'],
                                 linewidth=0.8, edgecolor='lightgray', facecolor='none', 
                                 linestyle='--', alpha=0.5)
        ax.add_patch(rect)
        # Word Text
        ax.text(row['x_min'], row['y_min'] - 2, row['WORD'].strip(), 
                fontsize=7, color='gray', family='serif')

    # 2. Raw Gaze (Drift Failure)
    ax.plot(df_seg['raw_x'], df_seg['raw_y'], color='red', linestyle='--', 
            marker='o', markersize=4, alpha=0.6, label='Raw Gaze (Drift)')
    
    # 3. Corrected Gaze (STOCK-T)
    ax.plot(df_seg['corrected_x'], df_seg['corrected_y'], color='green', linestyle='-', 
            marker='*', markersize=6, alpha=1.0, linewidth=2.0, label='STOCK-T (Corrected)')

    # Aesthetics
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.invert_yaxis() # Screen coordinates
    
    # Set plot limits to focus on the trajectory
    all_x = pd.concat([df_seg['raw_x'], df_seg['corrected_x'], df_seg.iloc[:PLOT_LIMIT]['true_x']])
    all_y = pd.concat([df_seg['raw_y'], df_seg['corrected_y'], df_seg.iloc[:PLOT_LIMIT]['true_y']])
    ax.set_xlim(all_x.min() - 20, all_x.max() + 20)
    ax.set_ylim(all_y.max() + 20, all_y.min() - 20) # Inverted
    
    ax.legend(loc='upper right', frameon=False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PDF_PATH, bbox_inches='tight')
    print(f"✅ NeurIPS Trajectory Rendered: {OUTPUT_PDF_PATH}")

if __name__ == "__main__":
    render_trajectory()
