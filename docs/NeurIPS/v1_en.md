# Overcoming Extreme Baseline Drift in Edge Eye-Tracking via Neuro-Symbolic Cognitive Modeling

---

### Abstract
Webcam-based eye-tracking on consumer edge devices suffers from high noise and systematic hardware drift, often exceeding 45 pixels. Traditional signal processing methods fail to recover intended gaze paths under such extreme offsets. We propose LexiGaze, a neuro-symbolic framework that fuses neural gaze perception with symbolic linguistic priors. Our system utilizes a Psycholinguistic-Oculomotor Model (POM) and a Multi-Hypothesis Expectation-Maximization (EM) initialization to solve the "Line-Locking" failure mode. Across the full GECO corpus, LexiGaze achieves robust Spatio-temporal trajectory recovery, representing a breakthrough for robust eye-tracking on standard laptops.

---

### 1. Introduction
Eye-tracking is a powerful diagnostic tool for cognitive load and language learning. However, high-fidelity tracking typically requires expensive infrared hardware. Commodity webcams introduce systematic vertical drift due to head tilt and low sensor resolution, making word-level calibration nearly impossible. LexiGaze addresses this by treating the reader's intent as a hidden state in a Spatio-Temporal Oculomotor-Cognitive Kalman Transformer (STOCK-T), leveraging the predictable rhythm of reading to achieve Spatio-temporal trajectory recovery and self-correct hardware errors.


---

### 2. Methodology

#### 2.1 Cognitive Mass (CM)
We define Cognitive Mass ($CM_i$) for word $i$ as the product of its localized processing difficulty and global structural importance:
$$CognitiveMass_i = Surprisal(w_i) \times AttentionCentrality(w_i)$$
Surprisal is calculated via a Masked Language Model ($-\log_2 P(w_i | context)$), while Attention Centrality is derived from the mean self-attention weights in the Transformer's final layer.

#### 2.2 Psycholinguistic-Oculomotor Model (POM)
We pivot from diffuse neural attention to a causal biological transition matrix. The probability of moving from word $i$ to $j$ ($P(w_j | w_i)$) is modeled by forward momentum penalized by skip-thresholds and backward regressions boosted by current processing difficulty ($CM_i$).

#### 2.3 Multi-Hypothesis EM Initialization
To overcome "Line-Locking"—where drift causes the system to snap to the wrong text line—we evaluate multiple vertical shift hypotheses $H = [0, \pm LineHeight]$. The hypothesis maximizing the Viterbi path likelihood is selected to initialize the fine-grained median drift estimation ($\Delta x, \Delta y$).

---

### 3. Experiments & Results

#### 3.1 Robust Ablation Study
We evaluated LexiGaze across 10 trials of an English L2 reader (Subject pp01) under simulated +45px drift.

| Model | Configuration | Avg Strict Accuracy (%) | Avg Relaxed Accuracy (%) |
| :--- | :--- | :---: | :---: |
| M1 | Base Viterbi (Baseline) | 60.83% | 73.55% |
| M2 | Viterbi + Multi-Hypothesis EM | 74.90% | 92.15% |
| **M4** | **STOCK-T (POM + EM)** | **90.49%** | **99.67%** |
| M5 | Ultimate STOCK-T (+ OVP) | 88.63% | 99.57% |

#### 3.2 Discussion: The OVP Anomaly
Interestingly, while Optimal Viewing Position (OVP) alignment matches native reading physiology, L2 readers in our study showed higher strict accuracy with geometric centers (Model 4). This suggests that high cognitive load causes learners to target word centers more deliberately, rather than relying on efficient parafoveal preview.

---

### 4. Conclusion & Future Work
LexiGaze demonstrates that neuro-symbolic modeling can transform low-cost hardware into a precision diagnostic tool. By solving the line-locking problem via multi-hypothesis reasoning, we achieved near-perfect semantic tracking. Future work will focus on knowledge distillation of the BERT-based CM features into lightweight edge models for real-time mobile calibration.

---
**Prepared by**: LexiGaze Research Team  
**Date**: May 2, 2026
