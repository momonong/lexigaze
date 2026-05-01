# Skill 9: Cognitive-Gated Sparse Attention Transition (STOCK-T v2)

## Context & Objective
The previous implementation of the Attention-Guided Matrix (Skill 8) caused an accuracy drop (from 49.4% to 35.9%) on linear readers. This happened because a static fusion weight ($\alpha = 0.5$) forced the Viterbi decoder to anticipate regressions even when the reader was reading fluently, and raw BERT attention is too dense/noisy compared to actual human saccades.

We need to upgrade the `AttentionGuidedMatrix` to use a **Cognitive-Gated Sparse Fusion** mechanism. This is the core novelty for our NeurIPS submission.

## Mathematical Formulation

### 1. Adaptive Alpha ($\alpha_i$)
Instead of a global $\alpha$, the fusion weight should be dynamically determined by the `Base_CM` (Cognitive Mass / Difficulty) of the *current* word $i$.
* If word $i$ is easy, $\alpha_i \to 1$ (rely entirely on physical forward momentum).
* If word $i$ is hard, $\alpha_i \to \text{min\_alpha}$ (allow attention-based regressions).
$$\alpha_i = 1.0 - (CM_i \times \text{regression\_sensitivity})$$
*(Clip $\alpha_i$ between [0.3, 1.0]).*

### 2. Attention Sparsification (Top-K Anchors)
Human regressions are sparse. We must filter the dense BERT attention matrix $P_{attn}$.
* For row $i$, keep only the Top-K (e.g., $K=2$) attention weights to preceding words ($j < i$).
* Set all other regression attention weights to $0$.
* Re-normalize the row.

### 3. Dynamic Fusion
For each row $i$:
$$T(i, j) = \alpha_i \cdot P_{phys}(j|i) + (1 - \alpha_i) \cdot SparseP_{attn}(j|i)$$

## Required Implementation
Modify `scripts/geco/core/attention_transition.py`:
1. Update `__init__` to accept `regression_sensitivity` (default 0.8) and `top_k_anchors` (default 2).
2. Modify `build_matrix(self, sentence_tokens, bert_attention_matrix, base_cm_array)`:
   - Note the new requirement: it must now take `base_cm_array` to calculate $\alpha_i$ per word.
   - Implement the Sparsification logic (zero-out non-top-K elements for $j<i$).
   - Implement the row-by-row Adaptive Alpha fusion.
   - Ensure the final matrix is row-normalized.

## Evaluation & Documentation (@docs)
* Update `SKILL_IMPLEMENTATION_LOG.md` for Skill 9.
* Write a short note in the docs explaining *why* this prevents the accuracy drop on L1 (fluent) readers while preserving regression tracking for L2 readers.