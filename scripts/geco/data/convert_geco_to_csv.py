import pandas as pd
import os

def convert_to_csv(file_path):
    print(f"⏳ Converting {file_path} to CSV...")
    csv_path = file_path.replace(".xlsx", ".csv")
    if os.path.exists(csv_path):
        print(f"✅ {csv_path} already exists.")
        return csv_path
    
    try:
        df = pd.read_excel(file_path, engine='calamine')
        df.to_csv(csv_path, index=False)
        print(f"✅ Converted {file_path} to {csv_path}")
        return csv_path
    except Exception as e:
        print(f"❌ Failed to convert {file_path}: {e}")
        # Fallback to openpyxl if calamine fails
        try:
            print("⏳ Falling back to openpyxl...")
            df = pd.read_excel(file_path, engine='openpyxl')
            df.to_csv(csv_path, index=False)
            print(f"✅ Converted {file_path} to {csv_path}")
            return csv_path
        except Exception as e2:
            print(f"❌ Fallback failed: {e2}")
            return None

if __name__ == "__main__":
    convert_to_csv("data/geco/L1ReadingData.xlsx")
    convert_to_csv("data/geco/L2ReadingData.xlsx")
