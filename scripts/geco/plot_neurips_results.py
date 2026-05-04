import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 設定 NeurIPS 學術圖表風格
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
sns.set_theme(style="whitegrid")

# 數據 (來自你的最新實驗)
models = ['STOCK-T\n(Edge-Optimized)', 'STOCK-T\n(Surprisal)', 'w/o POM\n(No Inertia)', 'w/o EM\n(Kalman)', 'w/o Temp\n(Pointwise)']
recovery_rates = [78.38, 67.57, 24.32, 0.00, 0.00]
top3_acc = [35.07, 26.27, 11.61, 9.37, 23.72]

# 顏色設定 (凸顯 Edge-Optimized)
colors_rec = ['#2ca02c', '#98df8a', '#ff7f0e', '#d62728', '#d62728']
colors_acc = ['#1f77b4', '#aec7e8', '#ff7f0e', '#d62728', '#d62728']

# --- 圖 1: Trajectory Recovery (抗漂移能力) ---
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(models, recovery_rates, color=colors_rec, edgecolor='black', linewidth=1.2)
ax.set_ylabel('Trajectory Recovery Rate (%)', fontweight='bold')
ax.set_title('Breaking the "Line-Locking" Failure Mode under +45px Drift', fontweight='bold', pad=15)
ax.set_ylim(0, 100)

# 加上數值標籤
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('fig_recovery_rate.pdf', dpi=300)
plt.close()

# --- 圖 2: OVP Washout Effect (Top-3 準確率對比) ---
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(models[:3], top3_acc[:3], color=colors_acc[:3], edgecolor='black', linewidth=1.2, width=0.6)
ax.set_ylabel('Top-3 Word Accuracy (%)', fontweight='bold')
ax.set_title('The "OVP Washout" Effect in High-Noise Edge Environments', fontweight='bold', pad=15)
ax.set_ylim(0, 45)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('fig_ovp_washout.pdf', dpi=300)
plt.close()

print("✅ 圖表生成完畢：fig_recovery_rate.pdf 與 fig_ovp_washout.pdf")