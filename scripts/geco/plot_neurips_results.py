import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 1. 自動建立 NeurIPS 圖片存放目錄
output_dir = os.path.join("docs", "NeurIPS", "figures")
os.makedirs(output_dir, exist_ok=True)

# 2. 全局學術圖表風格設定 (NeurIPS 風格)
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.autolayout': True,
    'hatch.linewidth': 1.0
})
# 白底＋淺灰 y-grid，比較接近頂會論文常見風格
sns.set_theme(style="whitegrid",
              rc={"font.family": "serif", "axes.grid.axis": "y"})

# 3. 實驗數據 (Data from Population Restoration Report)
models = [
    'STOCK-T\n(Edge-Opt)',
    'STOCK-T\n(Surprisal)',
    'w/o POM\n(No Inertia)',
    'w/o EM\n(Kalman)',
    'w/o Temp\n(Pointwise)'
]
strict_acc = [15.54, 11.93, 4.97, 3.24, 11.40]
top3_acc = [35.07, 26.27, 11.61, 9.37, 23.72]
recovery_rates = [78.38, 67.57, 24.32, 0.00, 0.00]

# 🎨 更偏 NeurIPS 的低飽和色盤（專業感＋主次分明）
color_ours = '#1F4E79'      # 深藍：主模型
color_contrast = '#2E8B57'  # 沉穩綠：主要對比模型
color_baseline = '#B0B7C3'  # 淺灰藍：其他 baseline

# 第一張圖的顏色陣列（主模型深藍，對比模型綠，其餘淡灰）
colors_rec = [
    color_ours,
    color_contrast,
    color_baseline,
    color_baseline,
    color_baseline
]

# =====================================================================
# Figure 1: Trajectory Recovery Rate (打破行鎖定 Line-Locking)
# =====================================================================
fig1, ax1 = plt.subplots(figsize=(8, 4.5))

bars1 = ax1.bar(
    models,
    recovery_rates,
    color=colors_rec,
    edgecolor='black',
    linewidth=0.9,
    width=0.6
)

ax1.set_ylabel('Trajectory Recovery Rate (%)', fontweight='bold')
ax1.set_title(
    'Breaking "Line-Locking": Recovery under Extreme Drift (+45px)',
    fontweight='bold',
    pad=15
)
ax1.set_ylim(0, 100)

# 在柱狀圖上方加入精確數值
for bar in bars1:
    yval = bar.get_height()
    if yval > 0:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 2,
            f'{yval:.1f}%',
            ha='center',
            va='bottom',
            fontweight='bold'
        )
    else:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            2,
            '0%',
            ha='center',
            va='bottom',
            color='darkred',
            fontweight='bold'
        )

fig1_path = os.path.join(output_dir, 'fig_trajectory_recovery.pdf')
plt.savefig(fig1_path, format='pdf', bbox_inches='tight', dpi=300)
print(f"✅ 成功生成圖表 1: {fig1_path}")
plt.close(fig1)

# =====================================================================
# Figure 2: The "OVP Washout" Effect (Strict vs. Top-3 Accuracy)
# =====================================================================
fig2, ax2 = plt.subplots(figsize=(9, 5))

x = np.arange(len(models))
width = 0.35

# Strict：深藍實心；Top-3：淺灰藍＋網底，強調兩種指標差異
rects1 = ax2.bar(
    x - width / 2,
    strict_acc,
    width,
    label='Strict Acc. (Pixel-perfect)',
    color='#1F4E79',
    edgecolor='black',
    linewidth=0.9
)
rects2 = ax2.bar(
    x + width / 2,
    top3_acc,
    width,
    label='Top-3 Acc. (Semantic)',
    color='#D0D6E2',
    edgecolor='black',
    linewidth=0.9,
    hatch='///'
)

ax2.set_ylabel('Accuracy (%)', fontweight='bold')
ax2.set_title(
    'Word-Level Decoding: Demonstrating the "OVP Washout" Effect',
    fontweight='bold',
    pad=15
)
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.legend(loc='upper right')
ax2.set_ylim(0, 45)

def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f'{height:.1f}%',
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'  # 數字加粗一點
        )

autolabel(rects1, ax2)
autolabel(rects2, ax2)

fig2_path = os.path.join(output_dir, 'fig_word_accuracy_comparison.pdf')
plt.savefig(fig2_path, format='pdf', bbox_inches='tight', dpi=300)
print(f"✅ 成功生成圖表 2: {fig2_path}")
plt.close(fig2)

print("🎉 所有 NeurIPS 專業圖表已輸出完畢！")