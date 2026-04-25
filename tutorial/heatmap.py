import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. 設定路徑與環境
# ==========================================
DATA_DIR = r"tutorial\data"
RAW_FILE = os.path.join(DATA_DIR, "my_eyes.csv")
CALIB_FILE = os.path.join(DATA_DIR, "calibrated_eyes.csv")
SAVE_DIR = r"tutorial\figures"
os.makedirs(SAVE_DIR, exist_ok=True)

# 假設螢幕解析度 (這會影響繪圖範圍)
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080 

def create_heatmap(csv_path, title, save_name, color_map="Reds"):
    if not os.path.exists(csv_path):
        print(f"⚠️ 找不到檔案: {csv_path}，跳過此圖。")
        return

    df = pd.read_csv(csv_path)
    
    # 建立畫布
    plt.figure(figsize=(12, 7))
    
    # 繪製熱力圖 (使用 Kernel Density Estimation)
    # x_px, y_px 是原始欄位；如果是校準後的，請確認欄位名稱是 vibe_x/vibe_y 或 x_px
    x_col = 'vibe_x' if 'vibe_x' in df.columns else 'x_px'
    y_col = 'vibe_y' if 'vibe_y' in df.columns else 'y_px'

    sns.kdeplot(
        data=df, x=x_col, y=y_col, 
        fill=True, thresh=0.05, levels=10, cmap=color_map, alpha=0.8
    )

    # 標示目標單字位置 (假設 phenomenon 在 450, 320)
    plt.scatter(450, 320, color='green', marker='*', s=300, label='Target Word')
    
    # 設定坐標軸範圍與方向 (模擬螢幕)
    plt.xlim(0, SCREEN_WIDTH)
    plt.ylim(0, SCREEN_HEIGHT)
    plt.gca().invert_yaxis() # 讓 (0,0) 在左上角
    
    plt.title(title, fontsize=16)
    plt.xlabel("Screen X (px)")
    plt.ylabel("Screen Y (px)")
    plt.legend()
    
    # 儲存圖片
    output_img = os.path.join(SAVE_DIR, save_name)
    plt.savefig(output_img)
    print(f"✅ 熱力圖已儲存至: {output_img}")
    plt.show()

# ==========================================
# 2. 執行繪圖 (對照組與實驗組)
# ==========================================
print("🎨 正在生成熱力圖對照表...")

# 原始數據熱力圖 (紅色系)
create_heatmap(RAW_FILE, "Heatmap: Raw Gaze Data (Neural Noise)", "heatmap_raw.png", "Reds")

# 校準後數據熱力圖 (藍色系)
create_heatmap(CALIB_FILE, "Heatmap: Neuro-Symbolic Calibrated Gaze", "heatmap_calibrated.png", "Blues")