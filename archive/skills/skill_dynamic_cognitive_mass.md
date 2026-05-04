# Skill 1: Dynamic Cognitive Field Generator

## Context & Objective
You are an expert AI researcher implementing a cognitive reading model for an Edge AI eye-tracking system. 
The goal is to upgrade our existing static `CognitiveMass` (CM) to a dynamic, time-decaying field. In human reading, once a word is foveated or previewed, its cognitive processing demand decreases over time.

## Mathematical Formulation
Instead of static $CM_i = Surprisal_i \times Attention_i$, implement the following dynamic update per time step $t$:
$$CM_i(t) = \text{Base\_CM}_i \times \exp(-\lambda \cdot E_i(t))$$
Where:
* $\text{Base\_CM}_i$: The original static cognitive mass for word $i$ (pre-calculated from BERT).
* $E_i(t)$: The cumulative "Exposure" of word $i$ up to time $t$.
* $\lambda$: A decay rate hyperparameter (default: 0.5).

Exposure updates based on the current noisy gaze $x_t$:
$$E_i(t) = E_i(t-1) + \mathcal{N}(x_t | \mu_i, \sigma^2)$$
Where $\mathcal{N}$ is a spatial Gaussian centered at word $i$'s pixel location $\mu_i$.

## Required Implementation
Create a Python class `DynamicCognitiveField`:
1.  **Initialize**: Takes a list of word bounding boxes and their pre-computed `Base_CM` scores.
2.  **Update Step**: Method `update(current_gaze_x, current_gaze_y)` that updates the exposure $E_i$ for all words and recalculates $CM_i(t)$.
3.  **Output**: Returns an array of current $CM$ probabilities for all words at time $t$.

## Constraints
* Keep computational overhead minimal (Edge AI friendly). Use NumPy vectorization.
* Ensure $CM_i(t)$ is normalized across the sentence at each time step.