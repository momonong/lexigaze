# Agent Skill: Cognitive Mass (CM) Extraction & Neuro-Symbolic Gaze Calibration

## 1. Skill Metadata
- **Project**: LexiGaze (A Neuro-Symbolic & Vibe Coding Workshop Portfolio)
- **Phase**: Phase 3 & Phase 4 Integration
- **Objective**: Extract "Cognitive Mass" (CM) from linguistic texts using Large Language Models (LLMs) and Symbolic NLP tools, and utilize it as a Bayesian Prior to correct webcam eye-tracking systemic drift (Vertical Drift).
- **Core Philosophy**: Neuro-Symbolic AI. Fusing raw neural perception (noisy WebGazer coordinates) with high-level symbolic cognition (language priors).

## 2. Theoretical Background
"Cognitive Mass" (CM) is a value between 0 and 1 that represents the localized "gravity" or "cognitive pause probability" a reader experiences on a specific word. Research indicates that readers (especially L2 learners) pause longer on words with high surprisal, deep syntactic complexity, or high Action Scores. By calculating CM, we can create a probabilistic map of where human eyes *should* land, which acts as a powerful prior for noisy edge-device calibration.

## 3. Mathematical Formulas & Expressions (The Algorithms)
This skill supports three interchangeable calculation modes for CM. All final CM values MUST be Min-Max normalized at the sentence level to `[8]`.

### Mode 1: Role-Conditioned Surprisal (Neural Prior)
Uses an LLM (e.g., Llama-3 or ModernBERT) prompted to act as an L2 learner (LLaSA framework) to calculate the negative log-likelihood (Surprisal).
- **Formula**: `CM_1(w_i) = MinMax( -log P(w_i | w_{<i}, Prompt_{L2}) )`
- **Implementation Note**: Pass a prompt like "Read this as an A2 English learner:" before the text. Extract the cross-entropy loss for each token.

### Mode 2: Layer-wise Action Score (Mechanistic Prior)
Adapts the "Action Score" from NLP difficulty estimation by summing the entropy across the model's layers (or epochs) to capture intrinsic task difficulty.
- **Formula**: `CM_2(w_i) = MinMax( sum_{l=1}^{L} H(P_l(w_i)) )`
- **Implementation Note**: Extract the hidden states or logits from intermediate layers of the Transformer. Calculate the Shannon Entropy `H` of the probability distribution at each layer, and sum them up.

### Mode 3: Multi-dimensional Linguistic Gravity (Neuro-Symbolic Prior - **Default**)
The most robust approach, mathematically fusing neural probabilities (LLM) with deterministic symbolic rules (Dependency Parser).
- **Formula**: `CM_3(w_i) = \alpha * S(w_i) + \beta * D_syntax(w_i) + \gamma * AoA(w_i)`
- **Variables**:
  - `S(w_i)`: Normalized Surprisal (Neural) from HuggingFace Transformers.
  - `D_syntax(w_i)`: Normalized Dependency Tree Depth (Symbolic) from `SpaCy`.
  - `AoA(w_i)`: Normalized Age of Acquisition (Lexical Database).
  - `\alpha, \beta, \gamma`: Tunable hyperparameters (default to 0.4, 0.4, 0.2).

## 4. Execution Pipeline (Vibe Coding Directives for AI Agent)

**Step A: Text Processing & Alignment**
1. Receive input text and tokenize it using BOTH the HuggingFace AutoTokenizer and SpaCy (`en_core_web_sm`).
2. Map LLM sub-word tokens back to SpaCy's word-level tokens to ensure visual coordinates can map to whole words.

**Step B: Cognitive Mass Calculation**
1. Implement a Python class `CognitiveMassCalculator`.
2. Based on the selected Mode (1, 2, or 3), compute the CM array for the sentence.
3. Output a structured JSON/Dictionary: `[{"word": "idiosyncrasy", "bounding_box": [x, y, w, h], "CM": 0.85}, ...]`

**Step C: Bayesian Gravity Snap (Phase 4 Fusion)**
Apply Bayes' Theorem to align noisy WebGazer.js coordinates (`s`) to the true target word (`t`).
- **Likelihood P(s|t)**: Modeled as a 2D Gaussian distribution centered on the word's bounding box. Calculates the geometric probability of the gaze landing on `s` if they meant to look at `t`.
- **Prior P(t)**: The Cognitive Mass `CM(t)` calculated in Step B.
- **Posterior P(t|s)**: `P(t|s) \propto P(s|t) * CM(t)`
- **Action**: Snap the noisy gaze coordinate `s` to the word `t` that maximizes the Posterior probability.

## 5. Required Python Libraries
- `transformers` (PyTorch)
- `spacy`
- `numpy`, `scipy.stats.multivariate_normal`
- `pandas` (for reading raw.csv and outputting calibrated.csv)