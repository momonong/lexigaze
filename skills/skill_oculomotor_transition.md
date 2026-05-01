# Skill 2: Oculomotor Transition Matrix Builder

## Context & Objective
You are implementing the oculomotor dynamics inspired by the SWIFT and E-Z Reader models. 
We need to build a transition probability matrix $P(w_j | w_i)$ for a Hidden Markov Model (HMM), where states are words in a sentence. This models where the eyes are likely to move next, given the current fixated word.

## Core Logic & Rules
Construct a transition matrix `T` of size $N \times N$ (where $N$ is sentence length).
1.  **Saccadic Momentum (Forward)**: Highest probability should be jumping to $i+1$ or $i+2$.
2.  **Word Skipping**: If word $i+1$ has a very low `Base_CM` (e.g., stop words like "the", "a"), increase the probability of skipping to $i+2$.
3.  **Regressions (L2 Learner Behavior)**: For second-language learners, regressions are common. If word $i$ has a high `Base_CM` (difficult word), assign a non-zero probability to jumping back to syntactically related previous words $j < i$ (you can use distance penalty $\exp(-|i-j|)$ for simplicity).

## Required Implementation
Create a Python module `transition_model.py` with a class `ReadingTransitionMatrix`:
1.  **Method**: `build_matrix(base_cm_array, is_L2_reader=False)`
2.  If `is_L2_reader=True`, increase the base probability weight for regressions (j < i).
3.  Apply softmax to ensure each row sums to 1.0.

## Output format
Return a NumPy 2D array representing the transition probabilities.