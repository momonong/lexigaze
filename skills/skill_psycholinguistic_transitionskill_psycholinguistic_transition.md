# Skill 10: Psycholinguistic-Oculomotor Transition Matrix (POM)

## Context & Objective
Our experiments with LLM Attention-guided transitions (Skills 8 & 9) revealed a significant accuracy drop (down to 33.3%). LLM attention is too diffuse and violates the causal, physical momentum of human reading. 
For our NeurIPS submission, we are pivoting to a purely mathematically driven **Psycholinguistic-Oculomotor Model (POM)** inspired by the SWIFT and E-Z Reader frameworks. 

The transition matrix $T$ will solely rely on word distances and the `Base_CM` (primarily driven by Surprisal) of the targets, completely removing BERT self-attention.

## Mathematical Formulation
We want to define the transition probability $P(w_j | w_i)$, moving from word $i$ to word $j$.

### 1. Base Physical Saccade (Forward Momentum)
Human eyes naturally prefer moving forward by approx. 1 word.
$$P_{fwd}(j | i) \propto \exp\\left( -\\frac{(j - (i + 1))^2}{2\\sigma_{fwd}^2} \\right) \\quad \\text{for } j > i$$

### 2. Cognitive Modulation (Surprisal/CM influence)
- **Skipping**: If word $j$ has a very low $CM$ (e.g., a short, predictable word like "the"), the probability of skipping it increases. Thus, we penalize the transition to $j$ if $CM_j$ is high.
- **Regressions**: If the current word $i$ has a very high $CM$ (reader is confused), the probability of moving backward ($j < i$) increases.
$$P_{reg}(j | i) \propto \exp\\left( -\\frac{|j - (i - 1)|}{\\sigma_{reg}} \\right) \\times CM_i \\quad \\text{for } j \\le i$$

### 3. POM Matrix Construction
For each row $i$:
1. Calculate the unnormalized scores for all $j$:
   $$Score(i, j) = \\begin{cases} 
   P_{fwd}(j | i) \\times (1 - \\gamma \\cdot CM_j) & \\text{if } j > i \\\\
   P_{reg}(j | i) & \\text{if } j \\le i
   \\end{cases}$$
   *(where $\\gamma$ is a scaling factor, e.g., 0.5, to ensure we don't zero out probabilities).*
2. Normalize the row using Softmax or simple sum division so it sums to 1.0.

## Required Implementation
1. Replace `AttentionGuidedMatrix` in `scripts/geco/core/transition_model.py` with the new `PsycholinguisticTransitionMatrix`.
2. The `build_matrix` method should only take `sentence_tokens` and `base_cm_array` (no more `bert_attention_matrix`).
3. Ensure this matrix is cleanly integrated back into the `viterbi_gaze_decode` pipeline along with the EM-AutoCal (Skill 6) and OVP (Skill 7).

## Output
Log the outcome in `SKILL_IMPLEMENTATION_LOG.md` as "Skill 10: POM Integration". Test this immediately on the GECO drift dataset to see if we can push past the 50.0% accuracy baseline established by the pure physical Viterbi.