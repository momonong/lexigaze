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
    csv_path = "data/geco/geco_pp01_bayesian_results.csv"
    if not os.path.exists(csv_path):
        return

    df_clean = pd.read_csv(csv_path).copy()

    required_cols = [
        "WORD",
        "WORD_ID",
        "WORD_TOTAL_READING_TIME",
        "webcam_x",
        "webcam_y",
        "calibrated_x",
        "calibrated_y",
    ]
    missing_cols = [c for c in required_cols if c not in df_clean.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for scanpath rendering: {missing_cols}")

    # ---- Decoupled architecture: static stimulus definition (NOT from chronological fixation logs) ----
    line1_words = ["There", "was", "a", "moment's", "stupefied", "silence."]
    line2_words = ["Japp", "cried,", "'you're", "the", "goods!"]

    def _layout_line(words, baseline_y, start_x=105.0, char_w=11.0, min_w=52.0, pad=16.0, gap=22.0):
        boxes = []
        cursor_x = start_x
        for word in words:
            width = max(min_w, len(word) * char_w + pad)
            center_x = cursor_x + width / 2.0
            boxes.append({
                "word": word,
                "x": center_x,
                "y": baseline_y,
                "w": width,
                "h": 34.0,
            })
            cursor_x += width + gap
        return boxes

    stimulus_boxes = _layout_line(line1_words, baseline_y=220.0) + _layout_line(line2_words, baseline_y=120.0)

    # ---- Real trajectory extraction from dataframe only ----
    def _normalize_word(token):
        cleaned = str(token).strip().lower()
        cleaned = cleaned.replace('"', "").replace("'", "")
        for ch in [",", ".", "!", "?", ";", ":"]:
            cleaned = cleaned.replace(ch, "")
        return cleaned

    target_vocab = {
        _normalize_word(w)
        for w in (line1_words + line2_words)
    }
    target_vocab.add("youre")

    word_ids = df_clean["WORD_ID"].astype(str)
    is_trial_3_5 = word_ids.str.startswith("3-5-")
    df_trial = df_clean.loc[is_trial_3_5].copy()
    df_trial["normalized_word"] = df_trial["WORD"].map(_normalize_word)

    # Restrict to the intended phrase region (There ... goods!)
    there_candidates = df_trial.index[df_trial["normalized_word"] == "there"]
    goods_candidates = df_trial.index[df_trial["normalized_word"] == "goods"]
    if len(there_candidates) == 0 or len(goods_candidates) == 0:
        raise ValueError("Could not locate phrase boundaries ('There' to 'goods!') in dataframe.")

    start_idx = there_candidates[0]
    end_idx = next((idx for idx in goods_candidates if idx >= start_idx), None)
    if end_idx is None:
        raise ValueError("Could not locate 'goods!' after 'There' in dataframe ordering.")

    df_phrase = df_trial.loc[start_idx:end_idx].copy()
    df_phrase = df_phrase[df_phrase["normalized_word"].isin(target_vocab)].copy()
    if df_phrase.empty:
        raise ValueError("No trajectory rows matched target Figure 4 phrase.")

    numeric_cols = ["webcam_x", "webcam_y", "calibrated_x", "calibrated_y", "WORD_TOTAL_READING_TIME"]
    for col in numeric_cols:
        df_phrase[col] = pd.to_numeric(df_phrase[col], errors="coerce")
    df_phrase = df_phrase.dropna(subset=numeric_cols)
    if df_phrase.empty:
        raise ValueError("Matched trajectory rows contain invalid numeric coordinates.")

    raw_x = df_phrase["webcam_x"].to_numpy()
    raw_y = df_phrase["webcam_y"].to_numpy()
    corrected_x = df_phrase["calibrated_x"].to_numpy()
    corrected_y = df_phrase["calibrated_y"].to_numpy()
    durations = df_phrase["WORD_TOTAL_READING_TIME"].to_numpy()

    dur_min = float(np.min(durations))
    dur_span = float(np.max(durations) - dur_min)
    if dur_span == 0.0:
        sizes = np.full_like(durations, 130.0, dtype=float)
    else:
        sizes = 70.0 + ((durations - dur_min) / dur_span) * 190.0

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Real trajectories (markers only, no chronological connecting lines)
    raw_points = ax.scatter(
        raw_x, raw_y, s=52, color="#E63946", marker="x",
        linewidth=1.4, alpha=0.9, zorder=2, label="Raw gaze"
    )
    corrected_points = ax.scatter(
        corrected_x, corrected_y, s=sizes, color="#2A9D8F", edgecolor="white",
        linewidth=0.6, alpha=0.95, zorder=3, label="STOCK-T corrected"
    )

    # Semantic gravity arcs: drifted raw -> corrected
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
                alpha=0.75,
                shrinkA=2,
                shrinkB=2,
                connectionstyle="arc3,rad=-0.2",
            ),
            zorder=2,
        )
    arc_proxy, = ax.plot([], [], color="gray", linestyle=":", linewidth=1.0, label="Semantic gravity arc")

    # Static stimulus boxes/text rendered on top for readability
    for box in stimulus_boxes:
        rect = patches.Rectangle(
            (box["x"] - box["w"] / 2.0, box["y"] - box["h"] / 2.0),
            box["w"],
            box["h"],
            linewidth=0.8,
            edgecolor="lightgray",
            facecolor="white",
            alpha=0.8,
            linestyle="--",
            zorder=5,
        )
        ax.add_patch(rect)
        ax.text(
            box["x"], box["y"], box["word"],
            ha="center", va="center", fontsize=10.5, family="serif", color="black", zorder=6
        )

    # Bounds from both stimulus and real trajectories
    stim_x = np.array([b["x"] for b in stimulus_boxes], dtype=float)
    stim_y = np.array([b["y"] for b in stimulus_boxes], dtype=float)
    all_x = np.concatenate([raw_x, corrected_x, stim_x])
    all_y = np.concatenate([raw_y, corrected_y, stim_y])
    ax.set_xlim(float(np.min(all_x)) - 55.0, float(np.max(all_x)) + 55.0)
    ax.set_ylim(float(np.min(all_y)) - 70.0, float(np.max(all_y)) + 45.0)

    # NeurIPS minimalist canvas
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(
        handles=[raw_points, corrected_points, arc_proxy],
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
