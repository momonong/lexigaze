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

def generate_fig5_scanpath():
    # 1. Canvas Setup
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 300)
    ax.invert_yaxis()  # Screen coordinates
    
    # Strictly NO axes, NO ticks, NO gridlines, NO borders
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 2. Mock Data & Text Setup
    line1_text = ["The", "unexpected", "consequence", "of", "the", "global", "phenomenon..."]
    line2_text = ["was", "felt", "across", "multiple", "technological", "sectors."]
    
    complex_words = ["unexpected", "consequence", "phenomenon"]
    
    line1_y = 100
    line2_y = 160
    start_x = 50
    word_spacing = 15
    char_width = 8
    
    words_data = []
    
    # Calculate positions for Line 1
    current_x = start_x
    for word in line1_text:
        w_width = len(word) * char_width + 10
        w_height = 30
        words_data.append({
            "word": word,
            "x": current_x,
            "y": line1_y - w_height/2,
            "w": w_width,
            "h": w_height,
            "center": (current_x + w_width/2, line1_y),
            "is_complex": word in complex_words
        })
        current_x += w_width + word_spacing
        
    # Calculate positions for Line 2
    current_x = start_x
    for word in line2_text:
        w_width = len(word) * char_width + 10
        w_height = 30
        words_data.append({
            "word": word,
            "x": current_x,
            "y": line2_y - w_height/2,
            "w": w_width,
            "h": w_height,
            "center": (current_x + w_width/2, line2_y),
            "is_complex": False
        })
        current_x += w_width + word_spacing

    # 3. Draw Bounding Boxes and Text
    for w in words_data:
        # Background highlight for complex words
        facecolor = "#FFD700" if w["is_complex"] else "none"
        alpha = 0.15 if w["is_complex"] else 0
        
        rect = patches.Rectangle((w["x"], w["y"]), w["w"], w["h"], 
                                 linewidth=1, edgecolor="lightgray", facecolor=facecolor, 
                                 linestyle="--", alpha=alpha if w["is_complex"] else 1)
        # If not complex, just the edge
        if not w["is_complex"]:
            rect.set_fill(False)
            
        ax.add_patch(rect)
        
        # Word Text
        ax.text(w["center"][0], w["center"][1], w["word"], 
                ha="center", va="center", family="serif", fontsize=10)

    # 4. Generate Trajectories (Focus on Line 1)
    line1_words = [w for w in words_data if w["word"] in line1_text]
    
    # Path: Corrected Gaze (Snapped to centers of Line 1)
    corrected_centers = [w["center"] for w in line1_words]
    # Add some variation in fixation duration (size)
    durations = [100, 250, 300, 150, 100, 200, 400] # ms
    sizes = [d/2 for d in durations]
    
    corrected_x = [c[0] for c in corrected_centers]
    corrected_y = [c[1] for c in corrected_centers]
    
    # Path: Raw Gaze (Drift downwards)
    # Starts near Line 1, drifts towards Line 2
    drift_y = np.linspace(0, 60, len(line1_words)) # Max drift 60px
    jitter_x = np.random.normal(0, 10, len(line1_words))
    jitter_y = np.random.normal(0, 10, len(line1_words))
    
    raw_x = [c[0] + jx for c, jx in zip(corrected_centers, jitter_x)]
    raw_y = [c[1] + dy + jy for c, dy, jy in zip(corrected_centers, drift_y, jitter_y)]
    
    # 5. Draw Correction Vectors (Arrows)
    for rx, ry, cx, cy in zip(raw_x, raw_y, corrected_x, corrected_y):
        ax.annotate("", xy=(cx, cy), xytext=(rx, ry),
                    arrowprops=dict(arrowstyle="->", color="gray", linestyle=":", lw=0.8, alpha=0.6))

    # 6. Plot Trajectories
    # Raw Gaze
    ax.plot(raw_x, raw_y, color="#E63946", linestyle="--", alpha=0.6, label="Raw Gaze (Hardware Drift)")
    ax.scatter(raw_x, raw_y, color="#E63946", marker="x", s=30, alpha=0.6)
    
    # STOCK-T Corrected
    # Emerald Green: #2A9D8F
    ax.plot(corrected_x, corrected_y, color="#2A9D8F", linewidth=2.5, alpha=0.9, label="STOCK-T Corrected")
    ax.scatter(corrected_x, corrected_y, color="#2A9D8F", s=sizes, marker="o", edgecolors="white", linewidths=0.5, alpha=0.9, zorder=5)

    # 7. Legend & Polish
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    
    # Title
    ax.text(400, 30, "Figure 5: Scanpath Trajectory & Line-Locking Correction", 
            ha="center", va="center", fontsize=12, fontweight="bold", family="serif")

    # Save
    output_dir = "docs/NeurIPS/figures"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fig_scanpath_correction_v2.pdf")
    plt.savefig(output_path, bbox_inches="tight")
    print(f"✅ Generated high-fidelity figure: {output_path}")

if __name__ == "__main__":
    generate_fig5_scanpath()
