# 🧠 IntelligentGaze: Neuro-Symbolic & Edge AI Workshop Project Overview

## 📍 專案現況 (Current Status)

本專案旨在展示 **Neuro-Symbolic AI (神經符號人工智慧)** 在邊緣運算 (Edge AI) 的應用。由於硬體限制 (如低成本 Webcam) 產生的數據通常帶有嚴重的雜訊與偏移，我們引入 Large Language Model (LLM) 提取的「語言先驗知識」來校準硬體感知的誤差。

### 核心工作流 (The 3-Phase Workflow)

| 階段 | 任務名稱 | 核心工具 | 產出檔案 |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **感知層 (Perception)** | `index.html` (WebGazer) | `data/my_eyes.csv` |
| **Phase 2** | **認知層 (Cognition)** | `text_model.py` (BERT Tiny) | `data/cognitive_weights.json` |
| **Phase 3** | **融合層 (Fusion)** | `calibrate.py` (Gravity Snap) | `data/calibrated_eyes.csv` |

---

### 📂 目錄架構詳解 (Directory Map)

- `tutorial/`
  - `index.html`: 前端眼動儀收集器。學生在此閱讀帶有難字的文本。
  - `text_model.py`: **神經符號引擎核心**。
    - 使用 `bert-tiny` 計算 Word Surprisal (驚奇度)。
    - 自動計算引力權重 ($\alpha$)。
  - `calibrate.py`: **學生 Vibe Coding 練習區**。
    - 預先載入資料，提供 Baseline (Moving Average)。
    - 挖空處讓學生嘗試「引力吸附演算法」。
  - `heatmap.py`: **視覺化成效驗證**。
    - 繪製 Raw vs. Calibrated 的 Kernel Density Estimation (KDE) 熱力圖。
  - `data/`: 存放 CSV 與 JSON 數據中間產物。
  - `figures/`: 存放視覺化結果 (`dashboard.png`, `heatmap_calibrated.png`)。

---

### 🛠️ 目前問題與痛點 (Pain Points)

1.  **引力半徑固定 (Static Radius)**：目前的引力吸附半徑固定在 150px，若眼動儀漂移 (Drift) 超過此範圍，校準就會失效。
2.  **時序連貫性不足**：吸附演算法目前是點對點操作，未考慮到閱讀時的掃視 (Saccade) 與注視 (Fixation) 的動態變化。
3.  **環境相依性**：`text_model.py` 目前依賴 GPU 環境與特定的 HuggingFace 模型路徑，在課堂環境下可能需要更輕量的本地部署方案。

---

### 🚀 未來發展方向 (Future Roadmap)

#### 1. 演算法進化：動態引力模型 (Dynamic Gravity Model)
- **概念**：根據注視時間 (Fixation Duration) 動態調整引力半徑。
- **目標**：解決硬體嚴重漂移時的吸附失敗問題。

#### 2. 邊緣部署優化：ONNX / TFLite 整合
- **概念**：將 Phase 2 的 BERT 模型轉換為 ONNX 格式，直接在前端 (JavaScript) 執行，實現完全脫離 Python 的瀏覽器內校準。

#### 3. 擴展場景：自適應 UI 佈局 (Adaptive Layout)
- **概念**：當系統發現用戶對某個區塊有高 Surprisal 且長時間注視時，自動放大字體或提供 Tooltip 輔助。

---
*Created by Gemini CLI - 2026/04/25*
