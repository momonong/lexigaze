import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_kinematics(file_path, label):
    df = pd.read_csv(file_path)
    
    # Calculate saccades
    df['dx'] = df['true_x'].diff()
    df['dy'] = df['true_y'].diff()
    df['dist'] = np.sqrt(df['dx']**2 + df['dy']**2)
    
    # Identify line breaks (large negative dx and positive dy)
    # Average line height is ~40-60px
    line_breaks = (df['dx'] < -200) & (df['dy'] > 20)
    
    # Forward saccades (non-line breaks, dx > 0)
    fwd_saccades = df[(df['dx'] > 0) & (~line_breaks)]['dx']
    
    # Regressions (non-line breaks, dx < 0)
    regressions = df[(df['dx'] < 0) & (~line_breaks)]['dx']
    
    # Durations
    durations = df['WORD_TOTAL_READING_TIME']
    
    stats = {
        "Label": label,
        "Avg Fwd Saccade (px)": fwd_saccades.mean(),
        "Max Fwd Saccade (px)": fwd_saccades.max(),
        "Regression Count": len(regressions),
        "Regression Rate (%)": (len(regressions) / len(df)) * 100,
        "Avg Fixation Duration (ms)": durations.mean(),
        "Total Words": len(df)
    }
    return stats, fwd_saccades, regressions, durations

if __name__ == "__main__":
    l2_file = "data/geco/geco_pp01_trial5_clean.csv"
    l1_file = "data/geco/geco_l1_pp01_trial5_clean.csv"
    
    l2_stats, l2_fwd, l2_reg, l2_dur = analyze_kinematics(l2_file, "L2 (English)")
    l1_stats, l1_fwd, l1_reg, l1_dur = analyze_kinematics(l1_file, "L1 (Dutch)")
    
    print("\n=== Kinematics Comparison (Subject pp01, Trial 5) ===")
    results = pd.DataFrame([l1_stats, l2_stats])
    print(results.to_string(index=False))
    
    # Plotting distributions
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(l1_fwd, alpha=0.5, label='L1', bins=20)
    plt.hist(l2_fwd, alpha=0.5, label='L2', bins=20)
    plt.title("Forward Saccade Length (px)")
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.hist(l1_dur, alpha=0.5, label='L1', bins=20)
    plt.hist(l2_dur, alpha=0.5, label='L2', bins=20)
    plt.title("Fixation Duration (ms)")
    plt.legend()
    
    plt.subplot(1, 3, 3)
    # Regression Magnitude (absolute value)
    plt.hist(np.abs(l1_reg), alpha=0.5, label='L1', bins=10)
    plt.hist(np.abs(l2_reg), alpha=0.5, label='L2', bins=10)
    plt.title("Regression Magnitude (px)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("docs/figures/l1_l2_kinematics.png")
    print("\n✅ Kinematics plot saved to docs/figures/l1_l2_kinematics.png")
