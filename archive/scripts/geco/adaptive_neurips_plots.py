import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from scipy.stats import linregress

# Ensure output directory exists
os.makedirs("docs/NeurIPS/figures", exist_ok=True)

# NeurIPS Golden Typography & Sizing Rules
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "pdf.fonttype": 42,   # Crucial for conference PDF embedding
    "ps.fonttype": 42
})
sns.set_palette("colorblind")

def save_fig(name):
    plt.savefig(f"docs/NeurIPS/figures/{name}.pdf", bbox_inches='tight')
    print(f"Saved: docs/NeurIPS/figures/{name}.pdf")

# --- Visualization 1: Noise Robustness (Line Chart) ---
def plot_noise_robustness():
    df = pd.read_csv("data/geco/noise_stress_results.csv")
    plt.figure(figsize=(5.5, 2.5))
    
    plt.plot(df['Drift'], df['Baseline'], marker='o', linestyle='--', label='Baseline (Nearest Box)')
    plt.plot(df['Drift'], df['EM_Only'], marker='s', linestyle='-.', label='EM Only (No POM)')
    plt.plot(df['Drift'], df['STOCK_T'], marker='D', linestyle='-', linewidth=2, label='STOCK-T (Ours)')
    
    plt.axvline(45, color='gray', linestyle=':', alpha=0.8, label='Std Hardware Drift')
    
    plt.xlabel("Systematic Vertical Drift (pixels)")
    plt.ylabel("Strict Word Accuracy (%)")
    plt.title("Noise Robustness Stress Test")
    plt.grid(True, axis='y', linestyle=':', alpha=0.5)
    plt.legend(loc='lower left')
    sns.despine()
    save_fig("noise_robustness_adaptive")

# --- Visualization 2: OVP Anomaly Correlation (Scatter Plot) ---
def plot_ovp_correlation():
    df = pd.read_csv("docs/experiments/full_corpus_ovp_results.csv")
    plt.figure(figsize=(3.5, 2.8))
    
    # Split by group
    l1 = df[df['Group'] == 'L1']
    l2 = df[df['Group'] == 'L2']
    
    plt.scatter(l1['Proficiency_Proxy'], l1['Delta_Acc'], color='blue', alpha=0.6, label='L1 (Native)', s=20)
    plt.scatter(l2['Proficiency_Proxy'], l2['Delta_Acc'], color='red', alpha=0.6, label='L2 (Bilingual)', s=20)
    
    # Trendline
    slope, intercept, r_value, p_value, std_err = linregress(df['Proficiency_Proxy'], df['Delta_Acc'])
    x_range = np.linspace(df['Proficiency_Proxy'].min(), df['Proficiency_Proxy'].max(), 100)
    plt.plot(x_range, slope * x_range + intercept, color='black', alpha=0.8, linestyle='-', label=f'r={r_value:.2f}')
    
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.xlabel("Avg Fixation Duration (ms)")
    plt.ylabel("$\Delta$ Acc (Center - OVP) (%)")
    plt.title("OVP Anomaly vs. Proficiency")
    plt.legend(fontsize=7)
    sns.despine()
    save_fig("ovp_correlation_adaptive")

# --- Visualization 3: Final Benchmark (Bar Chart) ---
def plot_final_benchmark():
    df = pd.read_csv("data/geco/geco_pp01_final_evaluation.csv")
    plt.figure(figsize=(5.5, 2.5))
    
    # Filter for key configurations to avoid clutter
    configs = ["Nearest Box", "Kalman Filter", "Static Bayesian", "Viterbi (Base)", "STOCK-T v2 (Final)"]
    df_filtered = df[df['Configuration'].isin(configs)].copy()
    
    bars = plt.bar(df_filtered['Configuration'], df_filtered['Accuracy (%)'], color=sns.color_palette("colorblind")[:len(df_filtered)])
    plt.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=8)
    
    plt.ylabel("Word-Level Accuracy (%)")
    plt.title("Ablation Study: Sequential Performance Gain")
    plt.xticks(rotation=15, ha='right')
    plt.ylim(0, 100)
    sns.despine()
    save_fig("final_benchmark_adaptive")

if __name__ == "__main__":
    plot_noise_robustness()
    plot_ovp_correlation()
    plot_final_benchmark()
