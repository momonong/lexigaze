import pandas as pd

# 原始檔案與輸出檔案路徑
input_file = "data\geco\L2ReadingData.xlsx"
output_file = "data\geco\geco_pp01_trial5_clean.csv"

print("⏳ 正在讀取原始 Excel 資料 (這可能需要一兩分鐘)...")
df = pd.read_excel(input_file)

print("🧹 正在進行資料過濾與清洗...")
# 1. 篩選特定受試者 (pp01) 與特定閱讀段落 (Trial 5)
df_filtered = df[(df['PP_NR'] == 'pp01') & (df['TRIAL'] == 5)].copy()

# 2. 濾除沒有注視座標的無效數據 (去除 '.' 的資料)
df_filtered = df_filtered[df_filtered['WORD_FIRST_FIXATION_X'] != '.']

# 3. 強制將座標轉換為數值型態
df_filtered['true_x'] = pd.to_numeric(df_filtered['WORD_FIRST_FIXATION_X'])
df_filtered['true_y'] = pd.to_numeric(df_filtered['WORD_FIRST_FIXATION_Y'])

# 4. 只保留我們演算法真正需要的「黃金欄位」，大幅縮小檔案體積
columns_to_keep = [
    'WORD_ID', 
    'WORD', 
    'true_x', 
    'true_y', 
    'WORD_TOTAL_READING_TIME' # 總閱讀時間 (可以用來佐證困難度)
]
df_clean = df_filtered[columns_to_keep]

# 5. 匯出成輕量化 CSV
df_clean.to_csv(output_file, index=False, encoding='utf-8')

print(f"✅ 過濾完成！已成功將 {len(df_clean)} 筆乾淨資料儲存至 '{output_file}'。")
print("預覽前 5 筆乾淨資料：")
print(df_clean.head())