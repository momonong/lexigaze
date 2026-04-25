import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 模組 1：資料讀取 (寫死，學生不動)
# ==========================================
try:
    # 加上 r 前綴，確保 Windows 路徑的反斜線不會報錯
    df = pd.read_csv('tutorial/data/my_eyes.csv')
    print(f"成功載入 {len(df)} 筆原始眼動數據！")
except FileNotFoundError:
    print("錯誤：找不到 my_eyes.csv")
    exit()

# ==========================================
# 模組 2：對照組 Baseline - 基礎移動平均 (已提供)
# ==========================================
# 這是業界最無腦的解法，讓學生看到數學的基礎力量
window_size = 5
df['baseline_x'] = df['x_px'].rolling(window=window_size, min_periods=1).mean()
df['baseline_y'] = df['y_px'].rolling(window=window_size, min_periods=1).mean()


# ==========================================
# 🌟 模組 3：VIBE CODING 優化區 (挖空給 AI)
# ==========================================
# 學生任務：請修改 knowledge_base.md，並讓 AI 在這裡寫出更強的校準演算法
# 目標：結合 NLP 先驗引力 (例如 'phenomenon' 座標 450, 320)，產出 'vibe_x' 與 'vibe_y'

# --- [AI 產出的程式碼貼在這裡] ---
# (為了防止報錯，預設先讓 vibe 等於 baseline)
df['vibe_x'] = df['baseline_x']
df['vibe_y'] = df['baseline_y']
# ---------------------------------


# ==========================================
# 模組 4：輸出校準結果 (資料工程解耦)
# ==========================================
output_path = r'tutorial\data\calibrated_eyes.csv'
# 輸出成新檔案，index=False 代表不輸出 DataFrame 的列編號
df.to_csv(output_path, index=False)
print(f"✅ 校準完成！包含神經符號特徵的資料已匯出至：{output_path}")


# ==========================================
# 模組 5：視覺化成效驗證 (寫死，學生不動)
# ==========================================
plt.figure(figsize=(12, 7))

# 1. 原始雜訊
plt.scatter(df['x_px'], df['y_px'], color='lightcoral', alpha=0.3, label='1. Raw Webcam Noise')

# 2. 對照組：無腦移動平均 (沒有認知科學邏輯)
plt.plot(df['baseline_x'], df['baseline_y'], color='orange', linewidth=1.5, linestyle='--', label='2. Baseline (Moving Average)')

# 3. 實驗組：Vibe Coding 產出的神經符號校準
plt.scatter(df['vibe_x'], df['vibe_y'], color='darkblue', alpha=0.8, label='3. Vibe-Coded Gaze (Neuro-Symbolic)')

# 標示引力目標 (這裡先寫死示範)
plt.scatter(450, 320, color='green', marker='*', s=300, edgecolor='black', zorder=5)
plt.text(465, 335, 'phenomenon (alpha=0.8)', fontsize=12, fontweight='bold', color='green')

plt.title('IntelligentGaze: Baseline vs. Vibe Coding Optimization', fontsize=16)
plt.xlabel('Screen X')
plt.ylabel('Screen Y')
plt.gca().invert_yaxis()
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig("tutorial/figures/result.png")
plt.show()