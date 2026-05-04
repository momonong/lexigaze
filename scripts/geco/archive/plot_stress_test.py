import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration
CSV_PATH = "data/geco/noise_stress_results.csv"
OUTPUT_PLOT_PATH = "docs/figures/noise_robustness_chart.png"

def plot_stress_test():
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: {CSV_PATH} not found. Run the stress test first.")
        return

    df = pd.read_csv(CSV_PATH)
    
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Plot Baseline
    plt.plot(df['Drift'], df['Baseline'], 'o--', color='red', label='Baseline (Nearest Box)', linewidth=1.5, markersize=6)
    
    # Plot EM Only
    plt.plot(df['Drift'], df['EM_Only'], 's--', color='orange', label='EM Only (No POM)', linewidth=1.5, markersize=6)
    
    # Plot STOCK-T
    plt.plot(df['Drift'], df['STOCK_T'], 'D-', color='blue', label='STOCK-T (Ours)', linewidth=2.5, markersize=8)
    
    # Styling
    plt.title('LexiGaze Gaze Tracking Robustness under Systematic Drift', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Systematic Vertical Drift (Pixels)', fontsize=12)
    plt.ylabel('Word-Level Strict Accuracy (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.xlim(-5, 80)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='lower left', frameon=True, shadow=True)
    
    # Annotate breakdown points
    # plt.annotate('Baseline Breakdown', xy=(30, df[df['Drift']==30]['Baseline'].values[0]), xytext=(40, 40),
    #              arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(OUTPUT_PLOT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PLOT_PATH)
    print(f"✅ Noise robustness chart saved to {OUTPUT_PLOT_PATH}")

if __name__ == "__main__":
    plot_stress_test()
