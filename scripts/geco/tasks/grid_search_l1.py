import os
import sys
import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.geco.core.transition_model import PsycholinguisticTransitionMatrix
from scripts.geco.core.viterbi_decoder import viterbi_gaze_decode
from scripts.geco.core.em_calibration import AutoCalibratingDecoder

# Configuration
DATA_PATH = "data/geco/geco_l1_pp01_cognitive_mass.csv"
DRIFT_Y = 45.0
SIGMA_X = 40.0
SIGMA_Y = 30.0

def inject_noise(df):
    np.random.seed(42)
    df['noisy_x'] = df['true_x'] + np.random.normal(0, SIGMA_X, len(df))
    df['noisy_y'] = df['true_y'] + np.random.normal(0, SIGMA_Y, len(df)) + DRIFT_Y
    return df

def evaluate_accuracy(target_words, predicted_words):
    correct = sum(1 for t, p in zip(target_words, predicted_words) if str(t).strip().lower() == str(p).strip().lower())
    return (correct / len(target_words)) * 100

def run_grid_search():
    df = pd.read_csv(DATA_PATH)
    df = inject_noise(df)
    
    word_boxes = [[row['true_x']-20, row['true_y']-15, row['true_x']+20, row['true_y']+15] for _, row in df.iterrows()]
    base_cm = df['cognitive_mass'].values
    gaze_sequence = df[['noisy_x', 'noisy_y']].values
    true_words = df['WORD'].values

    # Parameter grid
    sigma_fwd_range = [0.5, 0.8, 1.0, 1.2, 1.5]
    sigma_reg_range = [0.5, 1.0, 1.5, 2.0]
    gamma_range = [0.1, 0.3, 0.5, 0.7, 0.9]

    results = []
    
    combinations = list(product(sigma_fwd_range, sigma_reg_range, gamma_range))
    
    for sf, sr, g in tqdm(combinations, desc="Grid Searching L1 Params"):
        pom_builder = PsycholinguisticTransitionMatrix(sigma_fwd=sf, sigma_reg=sr, gamma=g)
        t_matrix = pom_builder.build_matrix(len(df), base_cm)
        
        calibrator = AutoCalibratingDecoder(calibration_window_size=30)
        # Using a faster decode without full Multi-Hypothesis probing for grid search speed
        # if the drift is known to be handled well by one hypothesis.
        # But for correctness, let's stick to the calibrator's logic.
        indices, _ = calibrator.calibrate_and_decode(gaze_sequence, word_boxes, base_cm, t_matrix, sigma_gaze=[SIGMA_X, SIGMA_Y])
        acc = evaluate_accuracy(true_words, [df.iloc[idx]['WORD'] for idx in indices])
        
        results.append({
            "sigma_fwd": sf,
            "sigma_reg": sr,
            "gamma": g,
            "accuracy": acc
        })

    res_df = pd.DataFrame(results)
    best = res_df.loc[res_df['accuracy'].idxmax()]
    
    print("\n" + "="*50)
    print("🏆 Best L1 (Native) Parameters Found:")
    print("="*50)
    print(best)
    print("="*50)
    
    res_df.to_csv("data/geco/l1_grid_search_results.csv", index=False)
    print("✅ Grid search results saved to data/geco/l1_grid_search_results.csv")

if __name__ == "__main__":
    run_grid_search()
