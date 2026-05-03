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
    # Preserved Lower Left legend as requested by user
    ax.legend(loc='lower left', ncol=2, frameon=True, facecolor='white', framealpha=0.9)
    sns.despine()
    save_fig('fig3_robustness')

# --- Fig 4: Qualitative Scanpath Recovery (Masterpiece - Decoupled Architecture) ---
def plot_scanpath_final():
    # 1. Extract REAL Coordinates
    csv_path = "data/geco/geco_pp01_bayesian_results.csv"
    if not os.path.exists(csv_path): return
    df_traj = pd.read_csv(csv_path).iloc[:6].copy()
    
    # 2. Dynamic Text Placement (Real X, Clean Y)
    clean_y = df_traj['true_y'].median()
    
    words = df_traj['WORD'].values
    real_x = df_traj['true_x'].values
    durations = df_traj['WORD_TOTAL_READING_TIME'].values
    
    raw_x = df_traj['webcam_x'].values
    raw_y = df_traj['webcam_y'].values
    corrected_x = df_traj['calibrated_x'].values
    corrected_y = df_traj['calibrated_y'].values
    
    # 4. Professional Rendering
    fig, ax = plt.subplots(figsize=(8, 3.5))
    
    # Pad limits based on real data
    all_x = np.concatenate([real_x, raw_x, corrected_x])
    all_y = np.concatenate([[clean_y], raw_y, corrected_y])
    ax.set_xlim(np.min(all_x) - 50, np.max(all_x) + 50)
    ax.set_ylim(np.min(all_y) - 50, np.max(all_y) + 50)
    
    # NO axes or spines
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    
    # Draw Background Stimulus (Z-order 1)
    for txt, x in zip(words, real_x):
        txt_str = str(txt).strip()
        w_width = len(txt_str) * 7.5 + 15
        
        # Subtle Highlight for keywords
        is_heavy = txt_str in ["stupefied", "silence.", "cried,"]
        fc = "#FF8C00" if is_heavy else "white"
        alpha_val = 0.2 if is_heavy else 1.0
        
        ax.add_patch(patches.Rectangle((x - w_width/2, clean_y - 16), w_width, 32, 
                                       lw=0.7, ec='lightgray', fc=fc, ls='--', alpha=alpha_val, zorder=1))
        ax.text(x, clean_y, txt_str, ha='center', va='center', fontsize=11, family='serif', zorder=1)

    # 3. Plot the REAL Trajectories
    # Raw Gaze (Z-order 2)
    ax.plot(raw_x, raw_y, color="#E63946", ls='--', alpha=0.35, lw=1.2, zorder=2, label="Raw Gaze (Hardware Drift)")
    ax.scatter(raw_x, raw_y, color="#E63946", marker='x', s=35, alpha=0.5, zorder=2)
    
    # Corrected Gaze (Z-order 4)
    ax.plot(corrected_x, corrected_y, color="#2A9D8F", lw=2.5, alpha=0.9, zorder=4, label="STOCK-T Corrected")
    # Marker sizes scaled by real GECO durations
    sizes = (durations / np.max(durations)) * 200 + 40
    ax.scatter(corrected_x, corrected_y, color="#2A9D8F", s=sizes, marker='o', ec='white', lw=0.6, alpha=0.95, zorder=4)

    # Semantic Gravity Arcs (Z-order 3)
    for i in range(len(real_x)):
        ax.annotate("", xy=(corrected_x[i], corrected_y[i]), xytext=(raw_x[i], raw_y[i]),
                    arrowprops=dict(arrowstyle="->", color="gray", linestyle=":", 
                                    shrinkA=2, shrinkB=2, alpha=0.5,
                                    connectionstyle="arc3,rad=-0.2"), zorder=3)
    ax.plot([], [], color="gray", ls=":", label="Semantic Gravity Arc", zorder=3)
    
    # Directional Flow on corrected trajectory
    for i in range(len(corrected_x)-1):
        ax.annotate("", xy=(corrected_x[i+1], corrected_y[i+1]), xytext=(corrected_x[i], corrected_y[i]),
                    arrowprops=dict(arrowstyle="-|>", color="#2A9D8F", lw=0, alpha=0.5, shrinkA=18, shrinkB=18),
                    zorder=4)

    # Legend (Strict positioning)
    ax.legend(bbox_to_anchor=(0.5, -0.15), loc='lower center', ncol=3, frameon=False, fontsize=10)
    
    plt.tight_layout()
    output_path = "docs/NeurIPS/figures/fig4_scanpath_recovery_masterpiece.pdf"
    plt.savefig(output_path, bbox_inches='tight', transparent=True)
    plt.savefig(output_path.replace('.pdf', '.png'), bbox_inches='tight') # PNG for quick review
    print(f"✅ Masterpiece Figure generated: {output_path}")

if __name__ == "__main__":
    print("🚀 Generating Final NeurIPS Figures (Decoupled Architecture)...")
    plot_kinematics()
    plot_ovp_anomaly()
    plot_robustness()
    plot_scanpath_final()
    print("\n🏁 All figures generated successfully.")
