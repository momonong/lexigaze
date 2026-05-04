# Skill 3: Spatio-Temporal Viterbi Gaze Decoder

## Context & Objective
Our current eye-tracking pipeline uses a static, point-by-point "Gravity Snap" which fails under extreme hardware drift (+45px). 
We are moving to a sequence-based decoding approach. We will use the Viterbi algorithm (or a banded HMM decoder) to find the most likely sequence of fixated words given a sequence of noisy gaze coordinates.

## Algorithm Specifications
We have an HMM setup:
* **Hidden States**: The actual word being read ($w \in \{1, 2, ..., N\}$).
* **Observations**: The noisy $(x, y)$ gaze coordinates from the webcam over time $T$.
* **Transition Probabilities $A$**: Provided by `ReadingTransitionMatrix` (Skill 2).
* **Emission Probabilities $B$**: The spatial distance probability combined with our `DynamicCognitiveField` (Skill 1).

At time $t$, the emission probability of observing gaze $g_t$ given state $w_i$ is:
$$P(g_t | w_i) \propto \mathcal{N}(g_t | \mu_i, \Sigma) \times CM_i(t)$$

## Required Implementation
Create a function `viterbi_gaze_decode(gaze_sequence, word_boxes, base_cm, transition_matrix)`:
1.  Initialize the DP table `V[N, T]` and path pointers.
2.  For $t = 1$ to $T$:
    * Update dynamic CM using the current gaze point.
    * Calculate emission probabilities for all states.
    * Perform the standard Viterbi step: $V[j, t] = \max_i (V[i, t-1] \cdot A[i, j]) \cdot B[j, t]$
3.  Backtrack to find the optimal word sequence.

## Constraints
* Optimize for speed. Since transitions are mostly local (banded matrix), you can limit the search space to $i \pm 3$ words to reduce $O(T N^2)$ complexity.