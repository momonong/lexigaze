import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
import os
import sys
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

# Ensure output directory exists
os.makedirs("docs/NeurIPS/figures", exist_ok=True)

# Add project root to path for core modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.geco.core.transition_model import PsycholinguisticTransitionMatrix
from scripts.geco.core.em_calibration import AutoCalibratingDecoder

# Global Typography & Style Constraints
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "pdf.fonttype": 42, # Crucial for NeurIPS embedding
    "ps.fonttype": 42
})
sns.set_palette("colorblind")

def save_fig(name):
    plt.savefig(f"docs/NeurIPS/figures/{name}.pdf", bbox_inches='tight')
    print(f"✅ Saved: docs/NeurIPS/figures/{name}.pdf")

# --- Figure 1: SWIFT-Style Cognitive Mass Heatmap ---
def plot_swift_cm_field():
    df = pd.read_csv("data/geco/geco_pp01_cognitive_mass.csv")
    # Take a sample segment (first 10 words)
    df_seg = df.iloc[:12].copy()
    
    fig, ax = plt.subplots(figsize=(5.5, 2.0))
    
    # Plot words
    for i, row in df_seg.iterrows():
        ax.text(row['true_x'], 1.2, row['WORD'].strip(), ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Generate 1D CM field
    x = np.linspace(df_seg['true_x'].min() - 50, df_seg['true_x'].max() + 50, 1000)
    y = np.zeros_like(x)
    
    for _, row in df_seg.iterrows():
        # Gaussian peak for each word weighted by CM
        peak = row['cognitive_mass'] * np.exp(-((x - row['true_x'])**2) / (2 * 25**2))
        y += peak
    
    # Plot the activation field
    ax.fill_between(x, 0, y, color='blue', alpha=0.2, label='Activation Field (CM)')
    ax.plot(x, y, color='blue', linewidth=1.5, alpha=0.8)
    
    ax.set_ylim(0, 1.8)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel("Spatio-Linguistic Field (x-axis)")
    ax.set_ylabel("Activation Intensity")
    ax.set_title("Cognitive Mass Activation Field (SWIFT-Style)")
    sns.despine(left=True, bottom=False)
    
    save_fig("fig_cm_activation_field")

# --- Figure 2: L1 vs L2 Psycholinguistic Effect Curve ---
def plot_psycholinguistic_effects():
    # Load cognitive mass data
    df_cm_l1 = pd.read_csv("data/geco/geco_l1_pp01_cognitive_mass.csv")
    df_cm_l2 = pd.read_csv("data/geco/geco_pp01_cognitive_mass.csv")
    
    # Load clean data for reading times
    df_clean_l1 = pd.read_csv("data/geco/geco_l1_pp01_trial5_clean.csv")
    df_clean_l2 = pd.read_csv("data/geco/geco_pp01_trial5_clean.csv")
    
    # Merge
    df_l1 = pd.merge(df_cm_l1, df_clean_l1[['WORD_ID', 'WORD_TOTAL_READING_TIME']], on='WORD_ID')
    df_l2 = pd.merge(df_cm_l2, df_clean_l2[['WORD_ID', 'WORD_TOTAL_READING_TIME']], on='WORD_ID')
    
    df_l1['Group'] = 'L1 (Native)'
    df_l2['Group'] = 'L2 (Bilingual)'
    
    df = pd.concat([df_l1, df_l2])
    
    # Filter for reasonable durations
    df = df[df['WORD_TOTAL_READING_TIME'] > 50]
    
    plt.figure(figsize=(3.5, 2.8))
    sns.regplot(data=df[df['Group'] == 'L1 (Native)'], x='cognitive_mass', y='WORD_TOTAL_READING_TIME', 
                scatter_kws={'alpha':0.3, 's':10}, label='L1 (Native)', color='blue')
    sns.regplot(data=df[df['Group'] == 'L2 (Bilingual)'], x='cognitive_mass', y='WORD_TOTAL_READING_TIME', 
                scatter_kws={'alpha':0.3, 's':10}, label='L2 (Bilingual)', color='red')
    
    plt.xlabel("Word Cognitive Mass")
    plt.ylabel("Fixation Duration (ms)")
    plt.title("Linguistic Mass vs. Cognitive Load")
    plt.legend()
    sns.despine()
    
    save_fig("fig_psycholinguistic_effects")

# --- Figure 3: Scanpath Trajectory & Line-Locking Correction ---
def plot_scanpath_correction():
    df = pd.read_csv("data/geco/geco_pp01_cognitive_mass.csv")
    
    # Simulation logic (Same as renderer script but refined aesthetics)
    np.random.seed(42)
    DRIFT_Y = 45.0
    SIGMA_X = 30.0
    SIGMA_Y = 20.0
    
    df['raw_x'] = df['true_x'] + np.random.normal(0, SIGMA_X, len(df))
    df['raw_y'] = df['true_y'] + np.random.normal(0, SIGMA_Y, len(df)) + DRIFT_Y
    
    word_boxes = [[row['true_x']-25, row['true_y']-15, row['true_x']+25, row['true_y']+15] for _, row in df.iterrows()]
    base_cm = df['cognitive_mass'].values
    gaze_sequence = df[['raw_x', 'raw_y']].values
    
    # Run STOCK-T
    pom_builder = PsycholinguisticTransitionMatrix(sigma_fwd=0.8, sigma_reg=1.5, gamma=0.3)
    t_matrix = pom_builder.build_matrix(len(df), base_cm)
    calibrator = AutoCalibratingDecoder(calibration_window_size=30)
    predicted_indices, drift = calibrator.calibrate_and_decode(gaze_sequence, word_boxes, base_cm, t_matrix, use_ovp=True)
    
    corrected_gaze = gaze_sequence - np.array(drift)
    
    # Visualization segment
    LIMIT = 15
    fig, ax = plt.subplots(figsize=(5.5, 1.8))
    
    # Draw Boxes
    for i in range(25): # Draw background context boxes
        row = df.iloc[i]
        rect = patches.Rectangle((row['true_x']-25, row['true_y']-15), 50, 30, 
                                 linewidth=0.5, edgecolor='gray', facecolor='none', linestyle='--', alpha=0.4)
        ax.add_patch(rect)
        ax.text(row['true_x'], row['true_y']-18, row['WORD'].strip(), ha='center', fontsize=6, color='gray')

    # Plot trajectories
    ax.plot(df['raw_x'][:LIMIT], df['raw_y'][:LIMIT], 'r--', marker='o', markersize=3, alpha=0.4, label='Raw Gaze (Drift)')
    ax.plot(corrected_gaze[:LIMIT, 0], corrected_gaze[:LIMIT, 1], 'g-', marker='s', markersize=4, linewidth=2, label='STOCK-T (Corrected)')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.legend(loc='upper right', frameon=False, fontsize=7)
    
    plt.title("Qualitative Scanpath Recovery (Breaking Line-Locking)")
    save_fig("fig_scanpath_correction")

# --- Figure 4: Hardware Stress Degradation ---
def plot_noise_degradation():
    df = pd.read_csv("data/geco/noise_stress_results.csv")
    
    plt.figure(figsize=(5.5, 2.0))
    plt.plot(df['Drift'], df['Baseline'], 'r--o', alpha=0.6, label='Spatial Baseline')
    plt.plot(df['Drift'], df['EM_Only'], 'g--s', alpha=0.6, label='EM (Physical Only)')
    plt.plot(df['Drift'], df['STOCK_T'], 'b-D', linewidth=2, label='STOCK-T (Neuro-Symbolic)')
    
    plt.axvline(45, color='black', linestyle=':', alpha=0.8)
    plt.text(46, 70, "Line-Height\nThreshold", fontsize=8, fontweight='bold')
    
    plt.xlabel("Vertical Drift (pixels)")
    plt.ylabel("Word Accuracy (%)")
    plt.ylim(0, 105)
    plt.title("Hardware Robustness Stress Test")
    plt.legend(loc='lower left', frameon=False)
    sns.despine()
    
    save_fig("fig_noise_degradation")

if __name__ == "__main__":
    plot_swift_cm_field()
    plot_psycholinguistic_effects()
    plot_scanpath_correction()
    plot_noise_degradation()
