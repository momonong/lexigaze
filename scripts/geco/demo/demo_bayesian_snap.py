import sys
import os
# Add the parent directory to sys.path to allow importing from the 'core' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from core.cm_algorithm import CognitiveMassCalculator, BayesianGravitySnap

# 1. Load Data
# We'll use the already preprocessed GECO data if available, or a subset
data_path = "data/geco/geco_pp01_trial5_clean.csv"
if not os.path.exists(data_path):
    print(f"❌ Data file {data_path} not found. Please run the data loader or check paths.")
    exit()

df = pd.read_csv(data_path)
full_text = " ".join(df['WORD'].astype(str).tolist())

# 2. Calculate Cognitive Mass (Neuro-Symbolic Mode)
calc = CognitiveMassCalculator()
cm_scores = calc.calculate_mode_3_neuro_symbolic(full_text)

# Align CM scores back to the dataframe
# Note: BERT tokenization (sub-words) usually results in more scores than raw words.
# We ensure the lengths match to avoid the ValueError.
min_len = min(len(df), len(cm_scores))
df = df.iloc[:min_len].copy()
df['CM'] = cm_scores[:min_len]

# 3. Simulate Noisy WebGaze
np.random.seed(42)
drift_y = 45 # 45px vertical drift
df['webcam_x'] = df['true_x'] + np.random.normal(0, 40, len(df))
df['webcam_y'] = df['true_y'] + np.random.normal(0, 30, len(df)) + drift_y

# 4. Bayesian Gravity Snap
snapper = BayesianGravitySnap(sigma_x=40, sigma_y=30)

# Prepare words data for the snapper
words_data = []
for _, row in df.iterrows():
    words_data.append({
        "word": row['WORD'],
        "pos": [row['true_x'], row['true_y']],
        "CM": row['CM']
    })

print("🧲 Starting Bayesian Gravity Snap...")

calibrated_results = []
for _, row in df.iterrows():
    gaze_pos = [row['webcam_x'], row['webcam_y']]
    best_match = snapper.snap(gaze_pos, words_data)
    
    calibrated_results.append({
        "calibrated_x": best_match['pos'][0],
        "calibrated_y": best_match['pos'][1],
        "snapped_word": best_match['word']
    })

cal_df = pd.DataFrame(calibrated_results)
df = pd.concat([df, cal_df], axis=1)

# 5. Evaluation
def is_accurate(x, y, tx, ty, threshold=30):
    return np.sqrt((x-tx)**2 + (y-ty)**2) <= threshold

df['raw_accurate'] = df.apply(lambda r: is_accurate(r['webcam_x'], r['webcam_y'], r['true_x'], r['true_y']), axis=1)
df['calibrated_accurate'] = df.apply(lambda r: is_accurate(r['calibrated_x'], r['calibrated_y'], r['true_x'], r['true_y']), axis=1)

print("\n" + "="*40)
print("📊 Bayesian Neuro-Symbolic Results")
print("="*40)
print(f"❌ Raw Webcam Accuracy: {df['raw_accurate'].mean()*100:.1f}%")
print(f"✅ Bayesian Calibrated Accuracy: {df['calibrated_accurate'].mean()*100:.1f}%")
print(f"🚀 Improvement: {(df['calibrated_accurate'].mean() - df['raw_accurate'].mean())*100:.1f}%")
print("="*40)

# Save results
output_path = "data/geco/geco_pp01_bayesian_results.csv"
df.to_csv(output_path, index=False)
print(f"Done! Results saved to {output_path}")
