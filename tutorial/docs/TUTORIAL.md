# LexiGaze: Neuro-Symbolic & Vibe Coding Workshop

## Laptop Setup
Prepare your environment before the session starts:

1. **Editor**: Use any editor you prefer (**VSCode**, **Cursor**, **Zed**, etc.). For minimalist enthusiasts, **Vim** or **Nano** are also perfectly fine.
2. **Google Account**: Required for accessing Google AI Studio and Colab. A standard free-tier account is sufficient.
3. **Environment**: Ensure you have Python 3.11+ installed. We recommend using a virtual environment.

## Getting Started

```bash
# 1. 下載專屬的 tutorial 分支
git clone -b tutorial/lets-goooo https://github.com/momonong/lexigaze.git

# 2. 進入專案資料夾
cd lexigaze

# 3. 安裝 Gemini CLI 並完成登入
npm install -g @google/gemini-cli
gemini login
```

## Overview
This workshop implements a Neuro-Symbolic AI architecture. We use **Neural Perception** (WebGazer) to capture gaze and **Symbolic Cognition** (LLM/BERT) to extract linguistic priors. These are fused via **Vibe Coding** to calibrate hardware errors.

## Prerequisites
Manage dependencies via `uv`, `conda`, or `pip`.

### Using uv
```bash
uv sync
```

### Using pip
```bash
pip install matplotlib pandas python-dotenv google-genai pdfplumber pyyaml
```

## Step 0: Environment Setup
Configure your Gemini API key for knowledge extraction.

1. Obtain a free API key from [Google AI Studio](https://aistudio.google.com/).
2. Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY="your_api_key_here"
   ```

## Step 1: Perception (Data Collection)
Capture raw, noisy gaze coordinates.

1. Start the collection server:
   ```bash
   python tutorial/server.py
   ```
2. Open `http://localhost:8000` in your browser.
3. Complete calibration and read the text provided.
4. Export data to `tutorial/data/raw.csv`.

## Step 2: Cognition (Prior Extraction)
Analyze linguistic difficulty to guide calibration.

### 2.1 Quantitative Analysis (Google Colab)
Compute "Word Surprisal" using `bert-tiny` (Torch-dependent).
1. Open the provided Colab notebook.
2. Run the BERT analysis on the target text.
3. Save the resulting `cognitive_weights.json` to `tutorial/data/`.

### 2.2 Skill Building (Local Knowledge Translation)
Extract symbolic rules from academic papers using Gemini.
1. Run the Skill Builder:
   ```bash
   python tutorial/skill_builder.py
   ```
2. Review extracted domain knowledge in `tutorial/skills/`.

## Step 3: Fusion (Neuro-Symbolic Calibration)
Fuse noisy perception with symbolic priors using Vibe Coding.

1. Execute the calibration engine:
   ```bash
   python tutorial/calibrate.py
   ```
   - Applies Moving Average (Baseline).
   - Implements "Gravity Snap" based on cognitive weights.
   - Outputs results to `tutorial/data/calibrated.csv`.

## Step 4: Verification (Visual Analytics)
Verify results via trajectory animation.

1. Generate gaze GIFs:
   ```bash
   python tutorial/visualize.py --target all
   ```
2. View animations in `tutorial/figures/`.

## Core Component Summary
- **server.py**: Local host for eye-tracking frontend.
- **skill_builder.py**: LLM-powered knowledge translator.
- **calibrate.py**: Fusion engine for perception and cognition.
- **visualize.py**: Gaze trajectory animator.

## Workshop Participants

| 組別 | 姓名 | 學號 | 系所 | 
| :--- | :--- | :--- | :--- | 
| 第一組 | 張盛文 | NM6134075 | 智慧科技碩士學程 |
| | 方滋堯 | NM6131069 | 智慧科技碩士學程 |
| | 蕭翔允 | U78111035 | 心理所 |
| | 陳姵蓉 | NP8121016 | 運算工程博士學程 |
| 第二組 | 彭成昊 | NM6141030 | 智慧科技碩士學程 |
| | 葉政晟 | L78131037 | 光電所 |
| | 古雲軒 | P76141576 | 資訊所 |
| | 宋容羽 | XX1142052 | 校際選課 |
| 第三組 | 林偉琦 | NM6131027 | 智慧科技碩士學程 |
| | 張襄翊 | NM6141022 | 智慧科技碩士學程 |
| | 郭勁恩 | NP8141024 | 運算工程博士學程 |
| 第四組 | 黃柏瑋 | NM6154067 | 智慧科技碩士學程 |
| | 孫明瀚 | NM6144020 | 智慧科技碩士學程 |
| | 盧提文 | NP8141016 | 運算工程博士學程 |
| | 鄭九彰 | NP8141579 | 運算工程博士學程 |
