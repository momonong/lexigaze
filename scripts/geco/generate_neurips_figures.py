import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def setup_neurips_style():
    """配置符合 NeurIPS 2026 標準的圖表參數"""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,          # 內文字體大小 10pt
        "axes.labelsize": 10,     # 軸標籤 10pt
        "axes.titlesize": 10,     # 標題 10pt
        "xtick.labelsize": 8,     # 刻度數字 8pt
        "ytick.labelsize": 8,
        "legend.fontsize": 8,     # 圖例 8pt
        "figure.dpi": 300,        # 300 DPI 解析度
        "axes.linewidth": 0.8,    # 邊框粗細
        "lines.linewidth": 1.5,   # 線條粗細
        "pdf.fonttype": 42,       # 確保 PDF 正確嵌入字體 (極度重要)
        "ps.fonttype": 42
    })
    sns.set_palette("colorblind") # 使用色盲友善配色

def main():
    setup_neurips_style()
    output_dir = "docs/NeurIPS/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    # 這裡由 Agent 根據實際 CSV 結構填入繪圖邏輯
    print(f"Reading data from data/geco/...")
    
    # --- Figure 1: EDA Stats (扁長型 5.5 x 2.0) ---
    fig1, ax1 = plt.subplots(figsize=(5.5, 2.0))
    # TODO: Load L1ReadingData.xlsx / L2ReadingData.xlsx and plot
    # ... (Agent will write the plot logic here)
    fig1.savefig(os.path.join(output_dir, "dataset_eda_stats.pdf"), bbox_inches='tight')
    plt.close(fig1)

    # --- Figure 2: Noise Robustness (扁長型 5.5 x 2.0) ---
    fig2, ax2 = plt.subplots(figsize=(5.5, 2.0))
    # TODO: Load data/geco/noise_stress_results.csv and plot lines
    # ... (Agent will write the plot logic here)
    fig2.savefig(os.path.join(output_dir, "noise_robustness_chart.pdf"), bbox_inches='tight')
    plt.close(fig2)

    # --- Figure 3: Qualitative Trajectory (扁長型 5.5 x 1.8) ---
    fig3, ax3 = plt.subplots(figsize=(5.5, 1.8))
    # TODO: Load data/geco/geco_l1_pp01_trial5_clean.csv and plot scatter/lines
    # ... (Agent will write the plot logic here)
    fig3.savefig(os.path.join(output_dir, "trial5_analysis.pdf"), bbox_inches='tight')
    plt.close(fig3)

    # --- Figure 4: OVP Correlation (近方形 3.5 x 2.8，適合放單欄文字旁) ---
    fig4, ax4 = plt.subplots(figsize=(3.5, 2.8))
    # TODO: Load geco_l1_final_evaluation.csv & geco_pp01_final_evaluation.csv
    # ... (Agent will write the plot logic here)
    fig4.savefig(os.path.join(output_dir, "ovp_proficiency_correlation.pdf"), bbox_inches='tight')
    plt.close(fig4)
    
    print(f"Successfully generated 4 NeurIPS PDF figures in {output_dir}/")

if __name__ == "__main__":
    main()