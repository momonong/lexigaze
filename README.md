# LexiGaze: Neuro-Symbolic & Vibe Coding Workshop

## Laptop Setup
Prepare your environment before the session starts:

1. **Editor**: Use any editor you prefer (**VSCode**, **Cursor**, **Zed**, etc.). For minimalist enthusiasts, **Vim** or **Nano** are also perfectly fine.
2. **Google Account**: Required for accessing Google AI Studio and Colab. A standard free-tier account is sufficient.
3. **Environment**: Ensure you have Python 3.11+ installed. We recommend using a virtual environment.

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
