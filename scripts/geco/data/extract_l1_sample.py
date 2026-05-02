import pandas as pd

# Path configuration
input_file = "data/geco/L1ReadingData.xlsx"
output_file = "data/geco/geco_l1_pp01_trial5_clean.csv"

print(f"⏳ Reading L1 Excel data from {input_file}...")
df = pd.read_excel(input_file)

print("🧹 Filtering and cleaning L1 data...")
# 1. Filter Subject (pp01) and Trial (5)
# Note: L1 might have different trial content than L2 even if IDs match.
df_filtered = df[(df['PP_NR'] == 'pp01') & (df['TRIAL'] == 5)].copy()

# 2. Filter invalid gaze coordinates
df_filtered = df_filtered[df_filtered['WORD_FIRST_FIXATION_X'] != '.']

# 3. Convert coordinates to numeric
df_filtered['true_x'] = pd.to_numeric(df_filtered['WORD_FIRST_FIXATION_X'])
df_filtered['true_y'] = pd.to_numeric(df_filtered['WORD_FIRST_FIXATION_Y'])

# 4. Keep core columns
columns_to_keep = [
    'WORD_ID', 
    'WORD', 
    'true_x', 
    'true_y', 
    'WORD_TOTAL_READING_TIME'
]
df_clean = df_filtered[columns_to_keep]

# 5. Export to CSV
df_clean.to_csv(output_file, index=False, encoding='utf-8')

print(f"✅ Extraction complete! Saved {len(df_clean)} records to '{output_file}'.")
print("Preview:")
print(df_clean.head())
