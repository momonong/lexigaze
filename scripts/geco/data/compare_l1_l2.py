import pandas as pd
import numpy as np

def inspect_geco_data(file_path, name):
    print(f"\n=== Inspecting {name} ({file_path}) ===")
    try:
        # Read only a few rows for inspection
        df = pd.read_excel(file_path, nrows=5)
        print(f"Columns: {df.columns.tolist()}")
        
        # Get sheet names
        xl = pd.ExcelFile(file_path)
        print(f"Sheet names: {xl.sheet_names}")
        
        # Get some basic info from the first sheet
        df_full = pd.read_excel(file_path, usecols=['PP_NR', 'TRIAL', 'WORD_ID', 'WORD_SKIP'])
        print(f"Number of rows: {len(df_full)}")
        print(f"Unique Subjects: {df_full['PP_NR'].nunique()}")
        print(f"Average Words per Trial: {df_full.groupby(['PP_NR', 'TRIAL'])['WORD_ID'].count().mean():.2f}")
        print(f"Global Skip Rate: {df_full['WORD_SKIP'].mean():.2%}")
        
    except Exception as e:
        print(f"Error inspecting {name}: {e}")

if __name__ == "__main__":
    l2_path = "data/geco/L2ReadingData.xlsx"
    l1_path = "data/geco/L1ReadingData.xlsx"
    
    inspect_geco_data(l2_path, "L2 Dataset")
    inspect_geco_data(l1_path, "L1 Dataset")
