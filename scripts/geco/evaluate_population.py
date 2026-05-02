import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.geco.core.transition_model import PsycholinguisticTransitionMatrix
from scripts.geco.core.viterbi_decoder import viterbi_gaze_decode
from scripts.geco.core.em_calibration import AutoCalibratingDecoder

# Noise Model
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

def run_subject_trial(lang, sub, trial, use_ovp):
    feat_path = f"data/geco/population/{lang}/{sub}/trial_{trial}/features.csv"
    if not os.path.exists(feat_path):
        return None
    
    df = pd.read_csv(feat_path)
    df = inject_noise(df)
    
    word_boxes = [[row['true_x']-20, row['true_y']-15, row['true_x']+20, row['true_y']+15] for _, row in df.iterrows()]
    base_cm = df['cognitive_mass'].values
    gaze_sequence = df[['noisy_x', 'noisy_y']].values
    true_words = df['WORD'].values
    
    # Use L2 optimized params for consistency in OVP analysis
    pom_builder = PsycholinguisticTransitionMatrix(sigma_fwd=0.8, sigma_reg=1.5, gamma=0.3)
    t_matrix = pom_builder.build_matrix(len(df), base_cm)
    
    calibrator = AutoCalibratingDecoder(calibration_window_size=30)
    indices, _ = calibrator.calibrate_and_decode(
        gaze_sequence, word_boxes, base_cm, t_matrix, 
        sigma_gaze=[SIGMA_X, SIGMA_Y], use_ovp=use_ovp
    )
    
    acc = evaluate_accuracy(true_words, [df.iloc[idx]['WORD'] for idx in indices])
    return acc

def main():
    subjects = ['pp01', 'pp02', 'pp03', 'pp04', 'pp05']
    trials = range(5, 11)
    languages = ['L1', 'L2']
    
    results = []
    
    for lang in languages:
        print(f"📂 Evaluating Group: {lang}")
        for sub in subjects:
            for trial in trials:
                acc_center = run_subject_trial(lang, sub, trial, use_ovp=False)
                acc_ovp = run_subject_trial(lang, sub, trial, use_ovp=True)
                
                if acc_center is not None:
                    results.append({
                        "Language": lang,
                        "Subject": sub,
                        "Trial": trial,
                        "Model 4 (Center)": acc_center,
                        "Model 5 (OVP)": acc_ovp
                    })
    
    res_df = pd.DataFrame(results)
    
    # Aggregation
    summary = res_df.groupby("Language").agg({
        "Model 4 (Center)": ["mean", "std"],
        "Model 5 (OVP)": ["mean", "std"]
    })
    
    print("\n📊 Population Accuracy Summary")
    print(summary)
    
    # Statistical Analysis (Paired t-test)
    report = "# Cross-Subject OVP Analysis (L1 vs. L2)\n\n"
    report += "## 1. Objective\nValidate if OVP alignment helps L1 (Native) readers and hinders L2 (Bilingual) readers.\n\n"
    report += "## 2. Results Table\n\n"
    report += "| Group | Model 4 (Center) Avg | Model 5 (OVP) Avg | Diff | p-value |\n"
    report += "| :--- | :---: | :---: | :---: | :---: |\n"
    
    for lang in languages:
        group_df = res_df[res_df['Language'] == lang]
        m4 = group_df['Model 4 (Center)'].values
        m5 = group_df['Model 5 (OVP)'].values
        t_stat, p_val = stats.ttest_rel(m4, m5)
        
        mean_m4 = np.mean(m4)
        mean_m5 = np.mean(m5)
        diff = mean_m5 - mean_m4
        
        report += f"| {lang} | {mean_m4:.2f}% | {mean_m5:.2f}% | {diff:+.2f}% | {p_val:.4f} |\n"
        print(f"\n{lang} t-test: p={p_val:.4f}")
    
    # Save Report
    os.makedirs("docs/experiments", exist_ok=True)
    with open("docs/experiments/2026-05-02_Cross_Subject_OVP_Analysis.md", "w") as f:
        f.write(report)
        
    print(f"\n✅ Analysis complete. Report saved to docs/experiments/2026-05-02_Cross_Subject_OVP_Analysis.md")

if __name__ == "__main__":
    main()
