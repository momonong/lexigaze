import pandas as pd
import numpy as np
import math

# 1. 載入帶有認知質量的資料
data_path = "data/geco/geco_pp01_cognitive_mass.csv"
print(f"⏳ 載入神經認知特徵: {data_path}")
df = pd.read_csv(data_path)

# 2. 定義核心演算法：事件視界 (Event Horizon) 公式
def calculate_gravity_radius(mass, r_base=25, r_max=120, alpha=0.5, beta=5.0):
    """計算單字的引力半徑"""
    sigmoid_weight = 1 / (1 + math.exp(-(alpha * mass - beta)))
    return r_base + (r_max - r_base) * sigmoid_weight

# 計算每個字的專屬引力半徑
df['gravity_radius'] = df['cognitive_mass'].apply(calculate_gravity_radius)

# 3. 模擬 Edge Webcam 的物理限制 (注入雜訊)
np.random.seed(42) # 固定亂數種子，確保實驗可重現
# 假設 Webcam 水平抖動 40px，垂直抖動 30px
df['noise_x'] = np.random.normal(0, 40, len(df))
df['noise_y'] = np.random.normal(0, 30, len(df))

# 系統性偏移 (Drift)：模擬使用者頭部稍微往下低，導致座標整體下移 45px (大約一行的高度)
drift_y = 45 

df['webcam_x'] = df['true_x'] + df['noise_x']
df['webcam_y'] = df['true_y'] + df['noise_y'] + drift_y

# 4. 執行 Neuro-Symbolic Gravity Snap (神經符號引力吸附)
def apply_gravity_snap(row, all_words_df):
    current_x = row['webcam_x']
    current_y = row['webcam_y']
    
    # 尋找周圍是否有引力場夠強的單字把游標吸過去
    for _, word_data in all_words_df.iterrows():
        # 計算 Webcam 座標與該單字中心的歐氏距離
        dist = math.sqrt((current_x - word_data['true_x'])**2 + (current_y - word_data['true_y'])**2)
        
        # 如果掉進引力事件視界 (距離小於該字的引力半徑)
        if dist <= word_data['gravity_radius']:
            # 觸發吸附！強制將座標拉回該單字的中心
            return word_data['true_x'], word_data['true_y'], True, word_data['WORD']
            
    # 如果沒被任何字吸住，就維持原本的錯誤座標
    return current_x, current_y, False, None

print("🧲 啟動神經符號引力場，開始校正座標...")
snap_results = df.apply(lambda row: apply_gravity_snap(row, df), axis=1)

# 將結果拆解存回 DataFrame
df['calibrated_x'] = [res[0] for res in snap_results]
df['calibrated_y'] = [res[1] for res in snap_results]
df['was_snapped'] = [res[2] for res in snap_results]
df['snapped_to_word'] = [res[3] for res in snap_results]

# 5. 論文關鍵指標 (Evaluation Metrics) 
# 定義：只要座標落在真實單字中心 30px 以內，就算「看對字」
def is_accurate(x, y, true_x, true_y, threshold=30):
    return math.sqrt((x - true_x)**2 + (y - true_y)**2) <= threshold

df['raw_accurate'] = df.apply(lambda r: is_accurate(r['webcam_x'], r['webcam_y'], r['true_x'], r['true_y']), axis=1)
df['calibrated_accurate'] = df.apply(lambda r: is_accurate(r['calibrated_x'], r['calibrated_y'], r['true_x'], r['true_y']), axis=1)

raw_accuracy = df['raw_accurate'].mean() * 100
calibrated_accuracy = df['calibrated_accurate'].mean() * 100

print("\n" + "="*40)
print("📊 實驗結果 (Neuro-Symbolic Calibration)")
print("="*40)
print(f"❌ 原始 Webcam 準確率: {raw_accuracy:.1f}% (因雜訊與飄移嚴重失準)")
print(f"✅ 引力校正後準確率: {calibrated_accuracy:.1f}%")
print(f"🚀 絕對提升率 (Absolute Improvement): {calibrated_accuracy - raw_accuracy:.1f}%")
print("="*40)

# 儲存最終比對數據，供後續畫圖使用
df.to_csv("data/geco/geco_pp01_final_evaluation.csv", index=False)