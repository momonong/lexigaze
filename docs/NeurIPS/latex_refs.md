# LaTeX Reference Metadata for LexiGaze NeurIPS Submission

## 1. Mathematical Formalisms

### Cognitive Mass (CM)
\begin{equation}
CognitiveMass_i = Surprisal(w_i) \times AttentionCentrality(w_i)
\end{equation}

### POM Transition Scores
\begin{equation}
Score(i, j) = \begin{cases} 
P_{fwd}(j | i) \times (1 - \gamma \cdot CM_j) & \text{if } j > i \\
P_{reg}(j | i) & \text{if } j \le i
\end{cases}
\end{equation}

### Regression Prior
\begin{equation}
P_{reg}(j | i) \propto \exp\left( -\frac{|j - (i - 1)|}{\sigma_{reg}} \right) \times CM_i \quad \text{for } j \le i
\end{equation}

---

## 2. Key Result Tables (LaTeX Format)

### Table 1: Population-Level Performance (GECO Full Corpus)
\begin{table}[h]
\centering
\begin{tabular}{lccc}
\hline
\textbf{Group} & \textbf{Strict Accuracy} & \textbf{Top-3 Accuracy} & \textbf{Trajectory Recovery} \\ \hline
L1 (Native) & 9.83\% & 19.82\% & 67.57\% \\
L2 (Bilingual) & 13.93\% & 32.38\% & 67.57\% \\ \hline
\end{tabular}
\caption{Global performance comparison across 37 subjects and ~3,200 trials under 45px vertical drift (Unbiased Baseline).}
\end{table}

### Table 2: Noise Robustness Stress Test (Subject pp01)
\begin{table}[h]
\centering
\begin{tabular}{cccc}
\hline
\textbf{Drift (px)} & \textbf{Baseline} & \textbf{EM Only} & \textbf{STOCK-T (Ours)} \\ \hline
0 & 32.34\% & 81.54\% & 90.49\% \\
30 & 24.58\% & 70.46\% & 90.49\% \\
45 & 19.10\% & 74.90\% & 90.49\% \\
60 & 13.42\% & 60.59\% & 82.50\% \\
75 & 8.50\% & 54.86\% & 51.95\% \\ \hline
\end{tabular}
\caption{Accuracy degradation across increasing levels of systematic vertical hardware drift.}
\end{table}

---

## 3. Figure Captions

- **Figure 1 (Architecture)**: System pipeline of LexiGaze showing the Spatio-Temporal Oculomotor-Cognitive Kalman Transformer (STOCK-T) flow from raw webcam input to calibrated gaze.
- **Figure 2 (Noise Robustness)**: Degradation curves showing STOCK-T's stability compared to spatial and physical baselines.
- **Figure 3 (OVP Correlation)**: Scatter plot showing the correlation between average fixation duration (proficiency) and the accuracy gain from geometric center-targeting ($\Delta_{Acc} = Acc_{Center} - Acc_{OVP}$).

---

## 4. Hyperparameter Locking
- $\sigma_{fwd} = 0.8$
- $\sigma_{reg} = 1.5$
- $\gamma = 0.3$
- $H = [0, 40, -40]$ px (EM Hypotheses)
