import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
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
    # 1. Canvas Setup
    # Adjusted height slightly to accommodate the legend outside
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 300)
    ax.invert_yaxis()  # Screen coordinates
    
    # Strictly NO axes, NO ticks, NO gridlines, NO borders
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 2. Corrected Text & Dynamic Spacing
    line1_text = ["The", "unexpected", "consequence", "of", "the", "global", "phenomenon"]
    line2_text = ["was", "felt", "across", "multiple", "technological", "sectors."]
    
    complex_words = ["unexpected", "consequence", "phenomenon"]
    
    line1_y = 120
    line2_y = 190
    start_x = 40
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

    # 4. Generate Trajectories (Focus on Line 1)
    line1_words = [w for w in words_data if w["line"] == 1]
    
    # Corrected Gaze (STOCK-T)
    corrected_centers = [w["center"] for w in line1_words]
    # Simulate fixation duration with marker size
    durations = [120, 320, 350, 180, 140, 240, 480]
    sizes = [d/2.5 for d in durations]
    
    cx = [c[0] for c in corrected_centers]
    cy = [c[1] for c in corrected_centers]
    
    # Raw Gaze (Drift downwards into Line 2 area)
    # Systematic drift + Jitter
    np.random.seed(42)
    drift_y_inc = np.linspace(0, 65, len(line1_words))
    rx = [c[0] + np.random.normal(0, 12) for c in corrected_centers]
    ry = [c[1] + dy + np.random.normal(0, 8) for c, dy in zip(corrected_centers, drift_y_inc)]

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
    
    # Directional Arrows on the corrected path
    for i in range(len(cx)-1):
        mid_x = (cx[i] + cx[i+1]) / 2
        mid_y = (cy[i] + cy[i+1]) / 2
        # Small arrow to show direction
        ax.annotate("", xy=(cx[i+1], cy[i+1]), xytext=(cx[i], cy[i]),
                    arrowprops=dict(arrowstyle="-|>", color="#2A9D8F", lw=0, alpha=0.6, shrinkA=15, shrinkB=15),
                    zorder=4)

    # 8. Legend Outside Plot
    ax.legend(bbox_to_anchor=(0.5, 1.12), loc='lower center', ncol=3, frameon=False, fontsize=9)
    
    # 9. Output
    output_path = "docs/NeurIPS/figures/fig_scanpath_correction_v2_final.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"✅ Final high-fidelity figure generated: {output_path}")

if __name__ == "__main__":
    generate_fig5_scanpath_final()
