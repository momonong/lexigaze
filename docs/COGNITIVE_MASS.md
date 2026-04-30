# 神經認知特徵萃取 (Cognitive Mass Extraction)

## 腳本目標 (Objective)

本腳本負責將清洗好的 GECO L2 (第二外語) 閱讀眼動數據，輸入至預訓練的語言模型中，計算每一個單字的**認知質量 (Cognitive Mass)**。這是本專案 "IntelligentGaze" 在論文中最重要的創新神經特徵。

## 論文核心：什麼是認知質量？

本腳本實作了論文方法學中的核心演算法，將以下兩個 NLP 指標融合為一個純數值：

1. **Surprisal (局部難度)**: $-\log_2 P(w_i | \text{context})$。由 BERT 的 MLM (Masked Language Modeling) 計算，代表單字出乎意料的程度。
2. **Attention Centrality (全局重要性)**: $\sum \text{Attention}(w_j \rightarrow w_i)$。提取 BERT 最後一層的注意力矩陣，代表該字在句法結構中的樞紐地位。

> **核心公式：** $CognitiveMass = Surprisal \times AttentionCentrality$
> *這個數值將在後續步驟中，轉化為物理空間中的「引力半徑 (Gravity Radius)」，用以吸附被 Webcam 雜訊污染的眼動座標。*

## 如何執行 (How to Run)

### 1. 準備輸入資料

確保你的目錄下有上一階段產出的乾淨資料集：`geco_pp01_trial5_clean.csv`。

### 2. 執行腳本

在擁有 GPU (例如 RTX 5090) 的環境下執行，速度最佳：

```bash
python scripts/geco/tasks/extract_cognitive_mass.py
```

