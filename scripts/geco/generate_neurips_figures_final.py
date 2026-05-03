import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np
import os
import sys
from scipy.stats import linregress

# Ensure project root in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.geco.core.transition_model import PsycholinguisticTransitionMatrix
from scripts.geco.core.em_calibration import AutoCalibratingDecoder

# --- NeurIPS Publication Standards ---
TEXT_WIDTH = 5.5
COLUMN_WIDTH = 3.5

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
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "text.usetex": False
})

def save_fig(name):
    plt.tight_layout()
    os.makedirs("docs/NeurIPS/figures", exist_ok=True)
    path = f"docs/NeurIPS/figures/{name}.pdf"
    plt.savefig(path, bbox_inches='tight', transparent=True)
    # Also save png for quick preview
    plt.savefig(f"docs/figures/{name}.png", bbox_inches='tight')
    print(f"✅ Saved Figure: {name}")

# --- Fig 1: Dataset Kinematics (L1 vs L2) ---
def plot_kinematics():
    # Data from Dataset EDA Report
    metrics = ['Fixation (ms)', 'Skip (%)', 'Regress (%)', 'Amp (words)']
    l1_vals = [287.8, 44.1, 30.7, 5.7]
    l2_vals = [354.3, 41.5, 32.5, 4.2]
    
    # Normalize by L1 for relative comparison
    l1_norm = np.array(l1_vals) / np.array(l1_vals)
    l2_norm = np.array(l2_vals) / np.array(l1_vals)
    
    # Increased height from 2.5 to 3.5 for better visual balance
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 3.5))
    y = np.arange(len(metrics))
    ax.barh(y + 0.2, l1_norm, 0.4, label='L1 (Native)', color='#4C72B0', alpha=0.9)
    ax.barh(y - 0.2, l2_norm, 0.4, label='L2 (Bilingual)', color='#C44E52', alpha=0.9)
    
    ax.set_yticks(y)
    ax.set_yticklabels(metrics)
    ax.set_xlabel('Relative Ratio (L1 = 1.0)')
    ax.set_title('Comparative Gaze Kinematics (GECO Corpus)')
    # Enabled frame with white background to avoid overlap
    ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9)
    ax.axvline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    sns.despine()
    save_fig('fig1_kinematics')

# --- Fig 2: OVP Anomaly Correlation ---
def plot_ovp_anomaly():
    csv_path = "docs/experiments/full_corpus_ovp_results.csv"
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 3.2), gridspec_kw={'width_ratios': [2, 1]})
    
    # Scatter plot
    l1 = df[df['Group'] == 'L1']
    l2 = df[df['Group'] == 'L2']
    ax1.scatter(l1['Proficiency_Proxy'], l1['Delta_Acc'], color='blue', s=20, alpha=0.5, label='L1')
    ax1.scatter(l2['Proficiency_Proxy'], l2['Delta_Acc'], color='red', s=20, alpha=0.5, label='L2')
    
    # Regression line
    slope, intercept, r, p, _ = linregress(df['Proficiency_Proxy'], df['Delta_Acc'])
    x = np.linspace(df['Proficiency_Proxy'].min(), df['Proficiency_Proxy'].max(), 100)
    ax1.plot(x, slope*x + intercept, 'k-', alpha=0.8, linewidth=1, label=f'r={r:.2f}')
    
    ax1.set_xlabel('Avg Fixation Duration (ms)')
    ax1.set_ylabel('$\Delta$ Accuracy (Center - OVP) (%)')
    ax1.set_title('(a) Proficiency Correlation')
    ax1.axhline(0, color='black', lw=0.5, ls='--')
    ax1.legend(frameon=False, fontsize=7)
    
    # Boxplot
    sns.boxplot(x='Group', y='Delta_Acc', data=df, ax=ax2, palette={'L1': 'blue', 'L2': 'red'}, width=0.6, fliersize=2)
    ax2.set_ylabel('')
    ax2.set_title('(b) Group Bias')
    ax2.axhline(0, color='black', lw=0.5, ls='--')
    
    sns.despine()
    save_fig('fig2_ovp_anomaly')

# --- Fig 3: Noise Robustness Stress Test ---
def plot_robustness():
    # Data from Noise Stress Test Report
    drifts = [0, 15, 30, 45, 60, 75]
    baseline = [32.34, 30.64, 24.58, 19.10, 13.42, 8.50]
    em_only = [81.54, 72.04, 70.46, 74.90, 60.59, 54.86]
    stock_t = [90.49, 90.75, 90.49, 90.49, 82.50, 51.95]
    
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 3.5))
    ax.plot(drifts, baseline, 'r--x', alpha=0.7, label='Spatial Baseline')
    ax.plot(drifts, em_only, 'g--s', alpha=0.7, label='EM (Physical Only)')
    ax.plot(drifts, stock_t, 'b-o', linewidth=2, label='STOCK-T (Ours)')
    
    ax.set_xlabel('Systematic Vertical Drift (pixels)')
    ax.set_ylabel('Strict Word Accuracy (%)')
    ax.set_title('Robustness to Hardware Degradation')
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.axvline(45, color='gray', linestyle=':', label='GECO Noise floor')
    # Changed to 2x2 legend (ncol=2) with background to avoid overlap
    ax.legend(loc='lower left', ncol=2, frameon=True, facecolor='white', framealpha=0.9)
    sns.despine()
    save_fig('fig3_robustness')

# --- Fig 4: Qualitative Scanpath Recovery (Real Data) ---
def plot_scanpath_final():
    # 1. Load Data
    cm_path = "data/geco/geco_pp01_cognitive_mass.csv"
    clean_path = "data/geco/geco_pp01_trial5_clean.csv"
    if not os.path.exists(cm_path): return
    
    df_cm = pd.read_csv(cm_path)
    df_clean = pd.read_csv(clean_path)
    df = pd.merge(df_cm, df_clean[['WORD_ID', 'WORD_TOTAL_READING_TIME']], on='WORD_ID')
    
    # Focus on first two lines (~22 words)
    LIMIT = 22
    df_seg = df.iloc[:LIMIT].copy()
    
    # Estimate boxes
    char_w = 7.0
    df_seg['w'] = df_seg['WORD'].str.strip().str.len() * char_w + 12
    df_seg['h'] = 28
    df_seg['x_min'] = df_seg['true_x'] - df_seg['w'] / 2
    df_seg['y_min'] = df_seg['true_y'] - df_seg['h'] / 2
    
    word_boxes = df_seg[['x_min', 'y_min', 'true_x', 'true_y']].copy() # Just for reference
    # Actual word_boxes for algorithm
    wb = [[r['x_min'], r['y_min'], r['x_min']+r['w'], r['y_min']+r['h']] for _, r in df_seg.iterrows()]
    
    # 2. Simulate Noise
    np.random.seed(42)
    drift_y = 45.0
    rx = df_seg['true_x'] + np.random.normal(0, 20, len(df_seg))
    ry = df_seg['true_y'] + np.random.normal(0, 12, len(df_seg)) + drift_y
    gaze_seq = np.stack([rx, ry], axis=1)
    
    # 3. Run Algorithm
    pom = PsycholinguisticTransitionMatrix(sigma_fwd=0.8, sigma_reg=1.5, gamma=0.3)
    tm = pom.build_matrix(len(df_seg), df_seg['cognitive_mass'].values)
    calibrator = AutoCalibratingDecoder(calibration_window_size=15)
    _, dv = calibrator.calibrate_and_decode(gaze_seq, wb, df_seg['cognitive_mass'].values, tm, use_ovp=True)
    
    corrected = gaze_seq - np.array(dv)
    cx, cy = corrected[:, 0], corrected[:, 1]
    
    # 4. Plot
    # Increased height from 3.8 to 4.2 to fit the title and legend comfortably
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    
    # Professional Title for NeurIPS standards
    ax.set_title('Qualitative Scanpath Recovery & Line-Locking Correction', pad=30, fontsize=12, fontweight='bold')
    
    # Background Stimulus
    cm_thresh = np.percentile(df_seg['cognitive_mass'], 75)
    for _, r in df_seg.iterrows():
        if r['cognitive_mass'] >= cm_thresh:
            ax.add_patch(patches.Rectangle((r['x_min']-2, r['y_min']-2), r['w']+4, r['h']+4, 
                                           facecolor="#FF8C00", alpha=0.1, zorder=1))
        ax.add_patch(patches.Rectangle((r['x_min'], r['y_min']), r['w'], r['h'], 
                                       lw=0.8, ec='lightgray', fc='none', ls='--', alpha=0.8, zorder=1))
        ax.text(r['true_x'], r['true_y'], r['WORD'].strip(), ha='center', va='center', fontsize=10, zorder=1)
        
    # Raw Gaze
    ax.plot(rx, ry, color="#E63946", ls='--', alpha=0.4, lw=1, zorder=2, label="Raw Gaze (Hardware Drift)")
    ax.scatter(rx, ry, color="#E63946", marker='x', s=20, alpha=0.5, zorder=2)
    
    # Gravity Arcs
    for i in range(len(df_seg)):
        ax.annotate("", xy=(cx[i], cy[i]), xytext=(rx[i], ry[i]),
                    arrowprops=dict(arrowstyle="->", color="gray", ls=":", shrinkA=3, shrinkB=3, alpha=0.4,
                                    connectionstyle="arc3,rad=-0.2"), zorder=3)
    ax.plot([], [], color="gray", ls=":", label="Semantic Gravity Arc", zorder=3)
    
    # Corrected Gaze
    ax.plot(cx, cy, color="#2A9D8F", lw=2, alpha=0.9, zorder=4, label="STOCK-T Corrected")
    sizes = (df_seg['WORD_TOTAL_READING_TIME'] / df_seg['WORD_TOTAL_READING_TIME'].max()) * 150 + 20
    ax.scatter(cx, cy, color="#2A9D8F", s=sizes, marker='o', ec='white', lw=0.5, alpha=0.9, zorder=4)
    
    # Directional Arrows
    for i in range(len(cx)-1):
        ax.annotate("", xy=(cx[i+1], cy[i+1]), xytext=(cx[i], cy[i]),
                    arrowprops=dict(arrowstyle="-|>", color="#2A9D8F", lw=0, alpha=0.4, shrinkA=12, shrinkB=12),
                    zorder=4)

    # Moved legend to bottom center to keep the top clear for the professional title
    ax.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=4, frameon=False, fontsize=9)
    save_fig('fig4_scanpath_recovery')

if __name__ == "__main__":
    print("🚀 Generating Final NeurIPS Figures (Real Data)...")
    plot_kinematics()
    plot_ovp_anomaly()
    plot_robustness()
    plot_scanpath_final()
    print("\n🏁 All figures generated successfully.")
