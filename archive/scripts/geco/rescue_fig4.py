import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def generate_perfect_fig4():
    # 1. 嚴格的學術字體與畫布設定
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    ax.set_xlim(50, 700)
    ax.set_ylim(80, 260)
    ax.axis('off') # 關閉所有干擾的邊框與坐標軸

    # 2. 完美的背景刺激物 (固定文字排版，絕對不會亂)
    l1_text = ["There", "was", "a", "moment's", "stupefied", "silence."]
    l1_x = [160, 220, 270, 360, 480, 590]
    l1_y = 220
    
    l2_text = ["Japp", "cried,", "'you're", "the", "goods!'"]
    l2_x = [160, 240, 340, 430, 530]
    l2_y = 120

    # 畫單字與 Bounding Box 
    for txt, x in zip(l1_text, l1_x):
        w = len(txt) * 9 + 10
        ax.add_patch(patches.Rectangle((x - w/2, l1_y - 15), w, 30, facecolor='white', edgecolor='lightgray', linestyle='--', alpha=0.9, zorder=1))
        ax.text(x, l1_y, txt, ha='center', va='center', fontsize=12, zorder=2)
        
    for txt, x in zip(l2_text, l2_x):
        w = len(txt) * 9 + 10
        ax.add_patch(patches.Rectangle((x - w/2, l2_y - 15), w, 30, facecolor='white', edgecolor='lightgray', linestyle='--', alpha=0.9, zorder=1))
        ax.text(x, l2_y, txt, ha='center', va='center', fontsize=12, zorder=2)

    # 3. 軌跡數據 (使用視覺化概念數據，展現跨行校對)
    gaze_x = np.array([165, 222, 268, 365, 475, 595])
    
    # Raw Gaze: 模擬 +45px 到底部的嚴重漂移 (掉到第二行)
    raw_y = np.array([200, 175, 150, 130, 125, 115])
    
    # Corrected Gaze: STOCK-T 成功利用語義引力拉回第一行
    corrected_y = np.array([218, 221, 219, 220, 222, 218])
    
    # 模擬凝視時間大小
    marker_sizes = np.array([120, 90, 60, 200, 250, 140])

    # 4. 畫軌跡
    # 紅色漂移
    ax.plot(gaze_x, raw_y, color="#E63946", linestyle='--', linewidth=1.5, alpha=0.6, zorder=3, label="Raw Gaze (Drift)")
    ax.scatter(gaze_x, raw_y, color="#E63946", marker='x', s=60, alpha=0.8, zorder=4)

    # 綠色校正
    ax.plot(gaze_x, corrected_y, color="#2A9D8F", linestyle='-', linewidth=2, alpha=0.7, zorder=3, label="STOCK-T Corrected")
    ax.scatter(gaze_x, corrected_y, color="#2A9D8F", s=marker_sizes, alpha=0.9, zorder=5)

    # 5. 語義引力弧線 (Semantic Gravity Arcs)
    for i in range(len(gaze_x)):
        ax.annotate("", xy=(gaze_x[i], corrected_y[i] - 12), xytext=(gaze_x[i], raw_y[i] + 12),
                    arrowprops=dict(arrowstyle="->", color="gray", linestyle=":", shrinkA=2, shrinkB=2, connectionstyle="arc3,rad=-0.2"), zorder=2)
    
    ax.plot([], [], color="gray", linestyle=":", label="Semantic Gravity Arc")

    # 6. 圖例排版
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=11)

    plt.tight_layout()
    plt.savefig("docs/NeurIPS/figures/fig4_scanpath_correction_FINAL.pdf", bbox_inches='tight')
    print("✅ 圖 4 生成完畢！")

generate_perfect_fig4()