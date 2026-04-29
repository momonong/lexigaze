import pandas as pd
import numpy as np
import os

class GecoDataLoader:
    def __init__(self, raw_data_path="data/geco/L2ReadingData.xlsx"):
        """
        初始化 DataLoader，如果找不到緩存，就會從原始 Excel 載入。
        """
        self.raw_data_path = raw_data_path
        self.df = None
        
    def load_raw_data(self):
        """載入龐大的原始 GECO L2 資料集 (53萬筆)"""
        if self.df is None:
            print(f"⏳ 載入原始 GECO 資料庫 ({self.raw_data_path})... 這可能需要幾分鐘。")
            self.df = pd.read_excel(self.raw_data_path)
            # 將座標轉換為數值，並把無效值 '.' 轉為 NaN
            self.df['WORD_FIRST_FIXATION_X'] = pd.to_numeric(self.df['WORD_FIRST_FIXATION_X'], errors='coerce')
            self.df['WORD_FIRST_FIXATION_Y'] = pd.to_numeric(self.df['WORD_FIRST_FIXATION_Y'], errors='coerce')
            print("✅ 原始資料庫載入完成！")
        return self.df

    def get_clean_trial(self, subject_id, trial_id):
        """
        萃取特定受試者與特定段落的乾淨資料，作為 Ground Truth。
        """
        df = self.load_raw_data()
        
        # 1. 篩選受試者與段落
        filtered = df[(df['PP_NR'] == subject_id) & (df['TRIAL'] == trial_id)].copy()
        
        # 2. 移除沒有座標的資料 (NaN)
        clean = filtered.dropna(subset=['WORD_FIRST_FIXATION_X', 'WORD_FIRST_FIXATION_Y']).copy()
        
        # 3. 重新命名為簡潔的欄位
        clean = clean.rename(columns={
            'WORD_FIRST_FIXATION_X': 'true_x',
            'WORD_FIRST_FIXATION_Y': 'true_y'
        })
        
        # 將閱讀時間強制轉為數字，遇到 '.' 就轉成 NaN (空值)
        clean['WORD_TOTAL_READING_TIME'] = pd.to_numeric(clean['WORD_TOTAL_READING_TIME'], errors='coerce')
        
        # 4. 只保留核心欄位
        columns_to_keep = ['WORD_ID', 'WORD', 'true_x', 'true_y', 'WORD_TOTAL_READING_TIME']
        
        print(f"📦 成功萃取 [{subject_id} - Trial {trial_id}]，共 {len(clean)} 筆有效注視點。")
        return clean[columns_to_keep]

# 測試區塊
if __name__ == "__main__":
    loader = GecoDataLoader()
    # 測試一次抓取多個 Trial
    df_t5 = loader.get_clean_trial("pp01", 5)
    df_t6 = loader.get_clean_trial("pp01", 6)
    
    # --- 新增這段來輸出資料 ---
    print("\n🔍 預覽 Trial 5 乾淨資料 (前 5 筆):")
    print(df_t5.head())
    
    print("\n📊 統計資訊:")
    print(f"Trial 5 總閱讀時間最長的前 3 個字:")
    print(df_t5.nlargest(3, 'WORD_TOTAL_READING_TIME')[['WORD', 'WORD_TOTAL_READING_TIME']])