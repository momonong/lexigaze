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

# --- Fig 4: Scanpath (two typographic rows + real per-word offset from true in px) ---
def plot_scanpath_final():
    csv_path = "data/geco/geco_pp01_trial5_stockt_from_main_m4_trajectory.csv"
    if not os.path.exists(csv_path):
        return

    plt.rcParams["font.family"] = "serif"

    line1_words = ["There", "was", "a", "moment's", "stupefied", "silence."]
    line2_words = ["Japp", "cried,", "'you're", "the", "goods!'"]
    line1_set, line2_set = frozenset(line1_words), frozenset(line2_words)
    ordered_words = line1_words + line2_words

    df = pd.read_csv(csv_path).copy()
    required_cols = [
        "WORD_ID", "WORD", "true_x", "true_y",
        "raw_x", "raw_y", "corrected_x", "corrected_y",
        "WORD_TOTAL_READING_TIME",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for Fig4 trajectory: {missing}")

    def _normalize_word(token):
        cleaned = str(token).strip().lower()
        for ch in ['"', "'", ",", ".", "!", "?", ";", ":"]:
            cleaned = cleaned.replace(ch, "")
        return cleaned

    ordered_norm_words = [_normalize_word(w) for w in ordered_words]

    df = df[df["WORD_ID"].astype(str).str.startswith("3-5-")].copy()
    df["norm_word"] = df["WORD"].map(_normalize_word)
    df = df.drop_duplicates(subset=["WORD_ID"], keep="first")

    selected = []
    used = set()
    for disp, norm_word in zip(ordered_words, ordered_norm_words):
        cand = df[(df["norm_word"] == norm_word) & (~df.index.isin(list(used)))]
        if cand.empty:
            continue
        idx = cand.index[0]
        used.add(idx)
        row = df.loc[idx].copy()
        row["display_word"] = disp
        selected.append(row)

    if len(selected) < 6:
        raise ValueError("Could not retrieve enough real trajectory points for the phrase.")

    df_phrase = pd.DataFrame(selected)
    for col in required_cols[2:]:
        df_phrase[col] = pd.to_numeric(df_phrase[col], errors="coerce")
    df_phrase = df_phrase.dropna(
        subset=["true_x", "true_y", "raw_x", "raw_y", "corrected_x", "corrected_y"]
    )
    if df_phrase.empty:
        raise ValueError("Selected rows have invalid coordinates.")

    x_lo, x_hi = 0.05, 0.95
    # Extra vertical gap so gaze markers (even after scaling) rarely sit on top of glyphs.
    y_line1, y_line2 = 1.22, -0.28

    words_line1 = [str(r["display_word"]) for _, r in df_phrase.iterrows() if str(r["display_word"]) in line1_set]
    words_line2 = [str(r["display_word"]) for _, r in df_phrase.iterrows() if str(r["display_word"]) in line2_set]

    def _pack_word_boxes(words, x0, x1, gap_min=0.014):
        """Return centers and half-widths (full width = 2*half_w) so boxes do not overlap."""
        n = len(words)
        if n == 0:
            return np.array([]), np.array([])
        half = np.array([0.018 + len(w) * 0.00545 for w in words], dtype=float)
        need = float(2.0 * np.sum(half) + gap_min * max(0, n - 1))
        span = float(x1 - x0)
        if need > span and need > 1e-9:
            half *= span / need * 0.98
        centers = np.empty(n, dtype=float)
        x_cursor = x0 + float(half[0])
        centers[0] = x_cursor
        for i in range(1, n):
            x_cursor += float(half[i - 1] + gap_min + half[i])
            centers[i] = x_cursor
        # Center the whole row in [x0, x1]
        row_half_span = float(centers[-1] + half[-1] - (centers[0] - half[0]))
        margin = 0.5 * (span - row_half_span)
        shift = x0 + margin - (centers[0] - half[0])
        centers = centers + shift
        return centers, half

    c1, half1 = _pack_word_boxes(words_line1, x_lo, x_hi)
    c2, half2 = _pack_word_boxes(words_line2, x_lo, x_hi)

    layout_x, layout_y, layout_half = [], [], []
    i1 = i2 = 0
    for _, row in df_phrase.iterrows():
        w = str(row["display_word"])
        if w in line1_set:
            layout_x.append(float(c1[i1]))
            layout_y.append(y_line1)
            layout_half.append(float(half1[i1]))
            i1 += 1
        else:
            layout_x.append(float(c2[i2]))
            layout_y.append(y_line2)
            layout_half.append(float(half2[i2]))
            i2 += 1
    lx = np.array(layout_x, dtype=float)
    ly = np.array(layout_y, dtype=float)
    lhalf = np.array(layout_half, dtype=float)

    tx = df_phrase["true_x"].to_numpy(dtype=float)
    ty = df_phrase["true_y"].to_numpy(dtype=float)
    dx_raw = df_phrase["raw_x"].to_numpy(dtype=float) - tx
    dy_raw = df_phrase["raw_y"].to_numpy(dtype=float) - ty
    dx_corr = df_phrase["corrected_x"].to_numpy(dtype=float) - tx
    dy_corr = df_phrase["corrected_y"].to_numpy(dtype=float) - ty

    def _axis_scale(*arrays, pct=88.0):
        """Robust |Δ| scale per axis so one outlier or one dominant axis does not squash the other."""
        stacked = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
        stacked = np.abs(stacked[np.isfinite(stacked)])
        if stacked.size == 0:
            return 1.0
        v = float(np.percentile(stacked, pct))
        return max(v, 1e-6)

    sx = _axis_scale(dx_raw, dx_corr, pct=88.0)
    sy = _axis_scale(dy_raw, dy_corr, pct=88.0)
    # Clip normalized offsets so one spiky word cannot paint over the whole row / other line.
    clip = 1.0
    nx_raw = np.clip(dx_raw / sx, -clip, clip)
    ny_raw = np.clip(dy_raw / sy, -clip, clip)
    nx_corr = np.clip(dx_corr / sx, -clip, clip)
    ny_corr = np.clip(dy_corr / sy, -clip, clip)
    band_x, band_y = 0.14, 0.12
    vis_boost = 1.08
    band_x *= vis_boost
    band_y *= vis_boost
    rx = lx + band_x * nx_raw
    ry = ly + band_y * ny_raw
    cx = lx + band_x * nx_corr
    cy = ly + band_y * ny_corr

    mean_raw = float(np.mean(np.hypot(dx_raw, dy_raw)))
    mean_corr = float(np.mean(np.hypot(dx_corr, dy_corr)))

    n_points = len(df_phrase)
    if df_phrase["WORD_TOTAL_READING_TIME"].notna().any():
        durs = df_phrase["WORD_TOTAL_READING_TIME"].fillna(
            df_phrase["WORD_TOTAL_READING_TIME"].median()
        ).to_numpy(dtype=float)
        d_min, d_max = float(np.min(durs)), float(np.max(durs))
        d_span = d_max - d_min
        sizes = np.full(n_points, 95.0) if d_span < 1e-6 else 55.0 + ((durs - d_min) / d_span) * 140.0
    else:
        sizes = np.full(n_points, 95.0)

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.95))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bh = 0.20
    split_idx = next(
        (i for i in range(n_points) if str(df_phrase.iloc[i]["display_word"]) in line2_set),
        n_points,
    )

    def _plot_scan_segments(xs, ys, **kwargs):
        """Do not draw one continuous polyline across both text rows (that reads as smear)."""
        label = kwargs.pop("label", None)
        if split_idx <= 0 or split_idx >= n_points:
            (ln,) = ax.plot(xs, ys, label=label, **kwargs)
            return ln
        (h1,) = ax.plot(xs[:split_idx], ys[:split_idx], label=label, **kwargs)
        ax.plot(xs[split_idx:], ys[split_idx:], label="_nolegend_", **kwargs)
        return h1

    raw_line = _plot_scan_segments(
        rx,
        ry,
        color="#C1121F",
        linestyle="--",
        linewidth=1.05,
        marker="x",
        markersize=6.0,
        markeredgewidth=1.1,
        alpha=0.72,
        zorder=3,
        label="Raw gaze (Δ vs true)",
    )
    corrected_line = _plot_scan_segments(
        cx,
        cy,
        color="#1b7f6a",
        linestyle="-",
        linewidth=1.35,
        marker="o",
        markersize=4.8,
        alpha=0.92,
        zorder=4,
        label="STOCK-T (Δ vs true)",
    )
    ax.scatter(
        cx, cy,
        s=sizes,
        color="#1b7f6a",
        edgecolor="white",
        linewidth=0.55,
        alpha=0.95,
        zorder=4,
    )

    for x0, y0, x1, y1 in zip(rx, ry, cx, cy):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->",
                color="#888888",
                linestyle=":",
                linewidth=0.95,
                alpha=0.55,
                connectionstyle="arc3,rad=-0.22",
                shrinkA=3,
                shrinkB=3,
            ),
            zorder=2,
        )

    for i in range(n_points):
        wtxt = str(df_phrase.iloc[i]["display_word"])
        bw = 2.0 * float(lhalf[i])
        rect = patches.Rectangle(
            (lx[i] - bw / 2.0, ly[i] - bh / 2.0),
            bw,
            bh,
            linewidth=0.75,
            edgecolor="#c0c0c0",
            facecolor="white",
            alpha=1.0,
            linestyle="--",
            zorder=12,
        )
        ax.add_patch(rect)
        ax.text(
            lx[i],
            ly[i],
            wtxt,
            ha="center",
            va="center",
            fontsize=9.0,
            family="serif",
            color="#1a1a1a",
            zorder=13,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.92),
        )
    arc_proxy, = ax.plot([], [], color="gray", linestyle=":", linewidth=1.0, label="Correction arc")

    pad = 0.07
    all_x = np.concatenate([rx, cx, lx])
    all_y = np.concatenate([ry, cy, ly])
    ax.set_xlim(float(np.min(all_x)) - pad, float(np.max(all_x)) + pad)
    ax.set_ylim(float(np.min(all_y)) - pad - 0.02, float(np.max(all_y)) + pad + 0.06)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    note = (
        f"Schematic word rows; gaze shows per-word offset from true center (px). "
        f"x scaled by p88|Δx|, y by p88|Δy|, ×{vis_boost:.2f} for legibility. "
        f"Mean ‖Δ‖: raw {mean_raw:.1f}px → corrected {mean_corr:.1f}px on this excerpt."
    )
    ax.text(
        0.5,
        -0.26,
        note,
        transform=ax.transAxes,
        fontsize=6.2,
        ha="center",
        va="top",
        color="#444444",
        linespacing=1.25,
    )

    ax.legend(
        handles=[raw_line, corrected_line, arc_proxy],
        bbox_to_anchor=(0.5, -0.12),
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=7.5,
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
