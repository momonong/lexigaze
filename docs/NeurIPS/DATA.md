# GECO L2 實驗數據萃取紀錄

## 數據來源
- **Dataset**: Ghent Eye-Tracking Corpus (GECO)
- **Subset**: Bilingual Reading Data (English L2)
- **Source File**: `L2ReadingData.xlsx`

## 萃取目標 (Sampling Strategy)
為了模擬真實的 ESL (第二外語) 學習者行為，同時確保演算法測試的即時性，我們採取了以下切片策略：
- **Subject ID**: `pp01` (荷蘭母語之英文學習者)
- **Trial ID**: `5` (包含複雜句構與低頻單字)
- **Data Points**: 156 筆有效注視記錄

## 清洗邏輯
1. **濾除 Skipping**: 移除了受試者跳過 (Word Skip) 的單字，僅保留真實產生注視座標 (`WORD_FIRST_FIXATION_X/Y`) 的資料。
2. **數值轉換**: 將原始資料中的座標字串轉化為數值型態 (Float)，並將螢幕座標標準化。
3. **特徵保留**: 保留 `WORD_TOTAL_READING_TIME` 作為後續驗證演算法與認知難度相關性的指標。

## 實驗目的
作為 **IntelligentGaze** 系統的 Ground Truth。我們將在此座標基礎上注入人為的高斯雜訊 (Gaussian Noise)，模擬邊緣設備 (Webcam) 的硬體限制，並測試「認知質量引力演算法」的修正準確率。