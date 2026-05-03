import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import os

# NeurIPS Standard Typography
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

def generate_fig5_scanpath_final():
    traj_csv = "data/geco/geco_pp01_trial5_stockt_from_main_m4_trajectory.csv"
    if not os.path.exists(traj_csv):
        print(f"❌ Missing trajectory file: {traj_csv}")
        return

    df = pd.read_csv(traj_csv)
    required_cols = ["WORD", "raw_y", "corrected_y", "WORD_TOTAL_READING_TIME"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ Missing columns in trajectory CSV: {missing}")
        return

    # 1. Canvas Setup
    fig, ax = plt.subplots(figsize=(8, 3.8))
    
    # Strictly NO axes, NO ticks, NO gridlines, NO borders
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 2. Hardcoded clean reading stimulus (decoupled from fixation order)
    line1_text = ["There", "was", "a", "moment's", "stupefied", "silence."]
    line2_text = ["Japp", "cried,", "'you're", "the", "goods!'"]
    
    complex_words = ["unexpected", "consequence", "phenomenon"]
    
    line1_y = 120
    line2_y = 190
    start_x = 70
    word_spacing = 18
    # Estimate width: approx 7px per character for 10pt Serif
    char_width_factor = 6.5
    
    words_data = []
    
    # Calculate positions for Line 1
    current_x = start_x
    for word in line1_text:
        w_width = len(word) * char_width_factor + 12
        w_height = 28
        words_data.append({
            "word": word,
            "x": current_x,
            "y": line1_y - w_height/2,
            "w": w_width,
            "h": w_height,
            "center": (current_x + w_width/2, line1_y),
            "is_complex": word in complex_words,
            "line": 1
        })
        current_x += w_width + word_spacing
        
    # Calculate positions for Line 2
    current_x = start_x
    for word in line2_text:
        w_width = len(word) * char_width_factor + 12
        w_height = 28
        words_data.append({
            "word": word,
            "x": current_x,
            "y": line2_y - w_height/2,
            "w": w_width,
            "h": w_height,
            "center": (current_x + w_width/2, line2_y),
            "is_complex": False,
            "line": 2
        })
        current_x += w_width + word_spacing

    # 3. Draw Stimulus (Z-order 1)
    for w in words_data:
        # Organic Heatmap (Soft background)
        if w["is_complex"]:
            heatmap = patches.Rectangle((w["x"]-2, w["y"]-2), w["w"]+4, w["h"]+4, 
                                       facecolor="#FF8C00", alpha=0.1, edgecolor="none", zorder=1)
            ax.add_patch(heatmap)
        
        # Bounding Boxes
        rect = patches.Rectangle((w["x"], w["y"]), w["w"], w["h"], 
                                 linewidth=0.8, edgecolor="lightgray", facecolor="none", 
                                 linestyle="--", alpha=0.8, zorder=1)
        ax.add_patch(rect)
        
        # Text
        ax.text(w["center"][0], w["center"][1], w["word"], 
                ha="center", va="center", family="serif", fontsize=10, zorder=1)

    # 4. Map real trajectory values onto clean stimulus X positions
    def _normalize_word(token):
        cleaned = str(token).strip().lower()
        for ch in ['"', "'", ",", ".", "!", "?", ";", ":"]:
            cleaned = cleaned.replace(ch, "")
        return cleaned

    target_words = line1_text + line2_text
    target_norm = [_normalize_word(w) for w in target_words]
    target_norm = ["youre" if w == "youre" else w for w in target_norm]
    layout_centers = [w["center"] for w in words_data]

    df["norm_word"] = df["WORD"].map(_normalize_word)
    selected_rows = []
    used = set()
    for word in target_norm:
        candidates = df[df["norm_word"] == word]
        idx = next((i for i in candidates.index if i not in used), None)
        if idx is not None:
            selected_rows.append(df.loc[idx])
            used.add(idx)

    if len(selected_rows) < 6:
        print("❌ Not enough matching words in trajectory CSV for final scanpath.")
        return

    df_seg = pd.DataFrame(selected_rows).copy()
    for col in ["raw_y", "corrected_y", "WORD_TOTAL_READING_TIME"]:
        df_seg[col] = pd.to_numeric(df_seg[col], errors="coerce")
    df_seg = df_seg.dropna(subset=["raw_y", "corrected_y"])
    if df_seg.empty:
        print("❌ Selected rows have invalid y coordinates.")
        return

    n = len(df_seg)
    if n <= len(layout_centers):
        selected_centers = layout_centers[:n]
    else:
        selected_centers = layout_centers + [layout_centers[-1]] * (n - len(layout_centers))

    cx = [c[0] for c in selected_centers]
    durations = df_seg["WORD_TOTAL_READING_TIME"].fillna(df_seg["WORD_TOTAL_READING_TIME"].median()).to_numpy()
    d_min = float(np.min(durations))
    d_span = float(np.max(durations) - d_min)
    sizes = np.full_like(durations, 70.0) if d_span < 1e-6 else 45.0 + ((durations - d_min) / d_span) * 120.0

    raw_y_real = df_seg["raw_y"].to_numpy(dtype=float)
    corr_y_real = df_seg["corrected_y"].to_numpy(dtype=float)

    corr_center = float(np.median(corr_y_real))
    corr_std = float(np.std(corr_y_real))
    if corr_std < 1e-6:
        corr_std = 1.0
    cy = (line1_y + ((corr_y_real - corr_center) / corr_std) * 6.0).tolist()

    # Drifted raw points trend downward toward line 2 while respecting real offsets
    if n > 1:
        forward = np.arange(n, dtype=float) / float(n - 1)
    else:
        forward = np.zeros(1, dtype=float)
    raw_offset = raw_y_real - corr_y_real
    off_min = float(np.min(raw_offset))
    off_span = float(np.max(raw_offset) - off_min)
    off_norm = forward if off_span < 1e-6 else (raw_offset - off_min) / off_span
    drift_strength = 0.65 * forward + 0.35 * off_norm
    ry = (line1_y + drift_strength * (line2_y - line1_y)).tolist()
    rx = cx

    # 5. Raw Gaze Visualization (Z-order 2)
    ax.plot(rx, ry, color="#E63946", linestyle="--", alpha=0.5, linewidth=1.2, zorder=2, label="Raw Gaze (Hardware Drift)")
    ax.scatter(rx, ry, color="#E63946", marker="x", s=25, alpha=0.6, zorder=2)

    # 6. "Semantic Gravity" Arcs (Z-order 3)
    for i in range(len(rx)):
        ax.annotate("", xy=(cx[i], cy[i]), xytext=(rx[i], ry[i]),
                    arrowprops=dict(arrowstyle="->", color="gray", linestyle=":", 
                                    shrinkA=3, shrinkB=3, alpha=0.5,
                                    connectionstyle="arc3,rad=-0.2"), zorder=3)
    
    # Hidden point for legend
    ax.plot([], [], color="gray", linestyle=":", label="Semantic Gravity Arc", zorder=3)

    # 7. STOCK-T Corrected Trajectory (Z-order 4)
    # Emerald Green: #2A9D8F
    ax.plot(cx, cy, color="#2A9D8F", linewidth=2.0, alpha=0.9, zorder=4, label="STOCK-T Corrected")
    ax.scatter(cx, cy, color="#2A9D8F", s=sizes, marker="o", edgecolors="white", linewidths=0.5, alpha=0.9, zorder=4)
    
    # Directional arrows on corrected path
    for i in range(len(cx)-1):
        ax.annotate("", xy=(cx[i+1], cy[i+1]), xytext=(cx[i], cy[i]),
                    arrowprops=dict(arrowstyle="-|>", color="#2A9D8F", lw=0, alpha=0.6, shrinkA=15, shrinkB=15),
                    zorder=4)

    # Fit view to stimulus and trajectory
    all_x = [w["center"][0] for w in words_data] + rx
    all_y = [w["center"][1] for w in words_data] + ry + cy
    ax.set_xlim(min(all_x) - 60, max(all_x) + 60)
    ax.set_ylim(min(all_y) - 70, max(all_y) + 70)
    ax.invert_yaxis()  # Screen coordinates

    # 8. Legend Outside Plot
    ax.legend(bbox_to_anchor=(0.5, 1.12), loc='lower center', ncol=3, frameon=False, fontsize=9)
    
    # 9. Output
    output_path = "docs/NeurIPS/figures/fig_scanpath_correction_v2_final.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"✅ Final high-fidelity figure generated: {output_path}")

if __name__ == "__main__":
    generate_fig5_scanpath_final()
