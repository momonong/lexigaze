import pandas as pd
import os
import glob

def analyze_file(file_path):
    print(f"--- Analyzing: {file_path} ---")
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            print(f"Unsupported format: {file_path}")
            return None
        
        summary = {
            "path": file_path,
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "sample": df.head(3).to_dict(orient='records')
        }
        return summary
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None

def main():
    data_files = []
    # Search patterns
    patterns = [
        "tutorial/data/*.csv",
        "data/geco/*.csv",
        "data/*.csv",
        "scripts/*/data/*.csv"
    ]
    
    for pattern in patterns:
        data_files.extend(glob.glob(pattern))
    
    results = []
    for f in data_files:
        res = analyze_file(f)
        if res:
            results.append(res)
            
    # Write to docs/DATASET_ANALYSIS.md
    with open("docs/DATASET_ANALYSIS.md", "w") as f:
        f.write("# Dataset Analysis Report\n\n")
        f.write(f"Analyzed {len(results)} data files.\n\n")
        
        for res in results:
            f.write(f"## File: {res['path']}\n")
            f.write(f"- **Rows**: {res['rows']}\n")
            f.write(f"- **Columns**: `{', '.join(res['columns'])}`\n")
            f.write("\n### Column Types\n")
            f.write("| Column | Type |\n| --- | --- |\n")
            for col, dtype in res['dtypes'].items():
                f.write(f"| {col} | {dtype} |\n")
            
            f.write("\n### Sample Data (First 3 rows)\n")
            f.write("```json\n")
            import json
            # Convert types for JSON
            sample_serializable = []
            for row in res['sample']:
                clean_row = {k: str(v) for k, v in row.items()}
                sample_serializable.append(clean_row)
            f.write(json.dumps(sample_serializable, indent=2))
            f.write("\n```\n\n")

    print(f"✅ Analysis complete. Report written to docs/DATASET_ANALYSIS.md")

if __name__ == "__main__":
    main()
