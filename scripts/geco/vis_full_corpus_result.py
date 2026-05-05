import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set NeurIPS style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300
})

def plot_ablation_results():
    # 1. 讀取數據
    df = pd.read_csv("data/geco/benchmark/full_corpus_results.csv")
    
    # 2. 準備作圖數據 (Recovery Rate)
    models = ['STOCK-T\n(Edge/Uniform)', 'STOCK-T\n(Surprisal)', 'w/o POM\n(No Inertia)', 'w/o EM\n(No Global Calib.)', 'Baseline\n(Spatial Only)']
    rec_means = [
        df['STOCK-T_Edge_Rec'].mean(),
        df['STOCK-T_Surprisal_Rec'].mean(),
        df['w/o_POM_Rec'].mean(),
        df['w/o_EM_Rec'].mean(),
        df['w/o_Temp_Rec'].mean()
    ]
    
    top3_means = [
        df['STOCK-T_Edge_Top3'].mean(),
        df['STOCK-T_Surprisal_Top3'].mean(),
        df['w/o_POM_Top3'].mean(),
        df['w/o_EM_Top3'].mean(),
        df['w/o_Temp_Top3'].mean()
    ]

    # 3. 畫圖 - 雙軸長條圖
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    width = 0.35

    # 畫 Recovery Rate (主指標)
    bars1 = ax1.bar(x - width/2, rec_means, width, label='Trajectory Recovery Rate (%)', color='#2b8cbe', edgecolor='black')
    ax1.set_ylabel('Recovery Rate (%)', color='#2b8cbe', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#2b8cbe')
    ax1.set_ylim(0, 100)

    # 畫 Top-3 Accuracy (副指標)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, top3_means, width, label='Top-3 Accuracy (%)', color='#f03b20', edgecolor='black')
    ax2.set_ylabel('Top-3 Accuracy (%)', color='#f03b20', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#f03b20')
    ax2.set_ylim(0, 50)

    # 標籤與細節
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_title('Ablation Study: Robustness against Extreme Hardware Drift (+45px)', pad=15, fontweight='bold')
    
    # 加上數值標籤
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

    # 合併 Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.tight_layout()
    plt.savefig('docs/NeurIPS/figures/fig_ablation_results.pdf', bbox_inches='tight')
    plt.show()

    # 4. 更深度的分析：L1 vs L2 (母語 vs 雙語) 的表現差異
    print("\n--- Deeper Analysis: L1 (Native) vs L2 (Bilingual) ---")
    l1_l2_stats = df.groupby('Lang')[['STOCK-T_Edge_Rec', 'w/o_POM_Rec']].mean().round(2)
    print(l1_l2_stats)

if __name__ == "__main__":
    plot_ablation_results()