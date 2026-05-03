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
    clean_csv_path = "data/geco/geco_pp01_trial5_clean.csv"
    bayes_csv_path = "data/geco/geco_pp01_bayesian_results.csv"
    if not os.path.exists(clean_csv_path) or not os.path.exists(bayes_csv_path):
        return

    segment_start = 75
    segment_end = 79

    df_clean = pd.read_csv(clean_csv_path).copy()
    df_bayes = pd.read_csv(bayes_csv_path).copy()
    df_clean["source_row"] = df_clean.index

    required_clean_cols = ["WORD_ID", "WORD", "true_x", "true_y", "WORD_TOTAL_READING_TIME", "source_row"]
    required_bayes_cols = ["WORD_ID", "webcam_x", "webcam_y", "calibrated_x", "calibrated_y"]
    missing_clean = [c for c in required_clean_cols if c not in df_clean.columns]
    missing_bayes = [c for c in required_bayes_cols if c not in df_bayes.columns]
    if missing_clean or missing_bayes:
        raise ValueError(
            f"Missing required columns. clean={missing_clean}, bayes={missing_bayes}"
        )

    df_segment = df_clean[(df_clean["source_row"] >= segment_start) & (df_clean["source_row"] <= segment_end)].copy()
    if df_segment.empty:
        raise ValueError(f"No rows found for requested segment {segment_start}->{segment_end}.")

    df_bayes = df_bayes.drop_duplicates(subset=["WORD_ID"], keep="first")
    df_segment = df_segment.merge(
        df_bayes[required_bayes_cols],
        on="WORD_ID",
        how="left",
    )

    numeric_cols = [
        "true_x", "true_y", "WORD_TOTAL_READING_TIME",
        "webcam_x", "webcam_y", "calibrated_x", "calibrated_y",
    ]
    for col in numeric_cols:
        df_segment[col] = pd.to_numeric(df_segment[col], errors="coerce")
    df_segment = df_segment.dropna(subset=["true_x", "true_y", "webcam_x", "webcam_y", "calibrated_x", "calibrated_y"])
    if df_segment.empty:
        raise ValueError("Segment rows are missing raw/corrected coordinates after merge.")

    # Extract data for this golden segment only
    word_text = df_segment["WORD"].astype(str).str.strip().to_numpy()
    word_center_x = df_segment["true_x"].to_numpy()
    raw_x = df_segment["webcam_x"].to_numpy()
    raw_y = df_segment["webcam_y"].to_numpy()
    corrected_x = df_segment["calibrated_x"].to_numpy()
    corrected_y = df_segment["calibrated_y"].to_numpy()
    durations = df_segment["WORD_TOTAL_READING_TIME"].fillna(df_segment["WORD_TOTAL_READING_TIME"].median()).to_numpy()

    # Clean baseline for static text stimulus
    baseline_y = float(np.median(df_segment["true_y"].to_numpy()))
    box_h = 34.0

    dur_min = float(np.min(durations))
    dur_span = float(np.max(durations) - dur_min)
    if dur_span == 0.0:
        marker_sizes = np.full_like(durations, 120.0, dtype=float)
    else:
        marker_sizes = 65.0 + ((durations - dur_min) / dur_span) * 150.0

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.25))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Trajectory lines + markers
    raw_line, = ax.plot(
        raw_x, raw_y,
        linestyle="--", linewidth=1.0, color="#E63946", alpha=0.5,
        marker="x", markersize=6.0, zorder=2, label="Raw gaze"
    )
    corrected_line, = ax.plot(
        corrected_x, corrected_y,
        linestyle="-", linewidth=1.1, color="#2A9D8F", alpha=0.85,
        marker="o", markersize=4.5, zorder=3, label="STOCK-T corrected"
    )
    ax.scatter(
        corrected_x, corrected_y,
        s=marker_sizes, color="#2A9D8F", edgecolor="white", linewidth=0.6,
        alpha=0.9, zorder=3
    )

    # Semantic gravity arrows: raw -> corrected
    for x0, y0, x1, y1 in zip(raw_x, raw_y, corrected_x, corrected_y):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->",
                color="gray",
                linestyle=":",
                linewidth=1.0,
                alpha=0.7,
                connectionstyle="arc3,rad=-0.2",
                shrinkA=2,
                shrinkB=2,
            ),
            zorder=2,
        )
    arc_proxy, = ax.plot([], [], color="gray", linestyle=":", linewidth=1.0, label="Semantic gravity arc")

    # Text and bounding boxes on top
    for txt, cx in zip(word_text, word_center_x):
        w = max(52.0, len(txt) * 10.0 + 16.0)
        rect = patches.Rectangle(
            (cx - w / 2.0, baseline_y - box_h / 2.0),
            w,
            box_h,
            linewidth=0.8,
            edgecolor="lightgray",
            facecolor="white",
            alpha=0.9,
            linestyle="--",
            zorder=5,
        )
        ax.add_patch(rect)
        ax.text(
            cx, baseline_y, txt,
            ha="center", va="center", fontsize=10.5, family="serif", color="black", zorder=6
        )

    all_x = np.concatenate([word_center_x, raw_x, corrected_x])
    all_y = np.concatenate([[baseline_y], raw_y, corrected_y])
    ax.set_xlim(float(np.min(all_x)) - 55.0, float(np.max(all_x)) + 55.0)
    ax.set_ylim(float(np.min(all_y)) - 60.0, float(np.max(all_y)) + 40.0)

    # No clutter
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(
        handles=[raw_line, corrected_line, arc_proxy],
        bbox_to_anchor=(0.5, -0.15),
        loc="lower center",
        ncol=3,
        frameon=False,
    )

    plt.tight_layout()
    output_path = "docs/NeurIPS/figures/fig4_scanpath_recovery_masterpiece.pdf"
    plt.savefig(output_path, bbox_inches="tight", transparent=False, facecolor="white")
    plt.savefig(output_path.replace(".pdf", ".png"), bbox_inches="tight", transparent=False, facecolor="white")
    print(f"✅ Masterpiece Figure generated: {output_path}")

if __name__ == "__main__":
    print("🚀 Generating Final NeurIPS Figures (Decoupled Architecture)...")
    plot_kinematics()
    plot_ovp_anomaly()
    plot_robustness()
    plot_scanpath_final()
    print("\n🏁 All figures generated successfully.")
