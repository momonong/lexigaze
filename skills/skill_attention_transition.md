# Skill 8: LLM Attention-Guided Oculomotor Transition Matrix

## Context & Objective
We are finalizing the Spatio-Temporal Oculomotor-Cognitive Kalman Transformer (STOCK-T) for a NeurIPS submission. We need to upgrade our rule-based Hidden Markov Model transition matrix to an LLM-guided dynamic matrix. 
Human readers often make saccadic regressions to syntactically related words when encountering difficulty. We will use the self-attention weights from a pre-trained language model (e.g., BERT) to predict these non-linear eye movements.

## Mathematical Formulation
The new transition matrix $T$ of size $N \times N$ (where $N$ is the sentence length) will be a weighted fusion of a "Physical Oculomotor Prior" and a "Linguistic Attention Prior".

For predicting the probability of moving from word $i$ to word $j$:

1.  **Physical Prior ($P_{phys}$)**: Models the mechanical forward momentum of reading (SWIFT model).
    $$P_{phys}(j|i) \propto \exp(-\beta \cdot |j - (i + \mu_{saccade})|)$$
    *(Where $\mu_{saccade}$ is typically $+1$ or $+1.5$, and $\beta$ controls the distance penalty).*

2.  **Attention Prior ($P_{attn}$)**: Models cognitive linking. Extracted from the final layers of BERT.
    $$P_{attn}(j|i) = \text{BERT\_Attention}(i, j)$$
    *(How much word $i$ attends to word $j$).*

3.  **Fusion (The Stock-T Transition)**:
    $$T(i, j) = \alpha \cdot P_{phys}(j|i) + (1 - \alpha) \cdot P_{attn}(j|i)$$
    *(Where $\alpha \in [0, 1]$ is a tunable hyperparameter, e.g., 0.7 for L1 readers, 0.5 for L2 readers who rely more on cognitive look-backs).*

## Required Implementation
Create a new module `scripts/geco/core/attention_transition.py` with a class `AttentionGuidedMatrix`.

1.  **Initialization**: `__init__(self, alpha=0.7, beta=0.5, mu_saccade=1.0)`
2.  **Method `build_matrix(self, sentence_tokens, bert_attention_matrix)`**:
    * Takes the list of tokens and the raw $N \times N$ attention matrix from BERT.
    * Calculates the Physical Prior matrix based on token distances.
    * Fuses it with the Attention matrix using the $\alpha$ parameter.
    * Applies row-wise normalization (sum to 1.0) to ensure it is a valid stochastic matrix for the Viterbi decoder.
3.  **Integration**: Update `viterbi_gaze_decode` to accept this new dynamic transition matrix instead of the old rule-based one.

## Output & Documentation (@docs)
* Output the clean, NumPy-vectorized code.
* Update the `SKILL_IMPLEMENTATION_LOG.md` to reflect the completion of Skill 8.
* Print a terminal output of a sample $10 \times 10$ transition matrix to verify that the probabilities correctly spike at $i+1$ (forward reading) but also show secondary spikes at syntactically related words.