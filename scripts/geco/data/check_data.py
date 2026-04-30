import pandas as pd

# 讀取 Excel 檔案
file_path = "data\geco\L2ReadingData.xlsx" 

print("⏳ 正在從 Excel 載入 GECO L2 資料集...")
# read_excel 會自動使用 openpyxl 引擎
df = pd.read_excel(file_path)

print(f"✅ 載入成功！資料列數：{len(df)}")
print("\n🔍 欄位清單：")
print(df.columns.tolist())

# 預覽前 5 筆，看看 WORD 和座標欄位
print("\n🔍 資料預覽：")
print(df.head())