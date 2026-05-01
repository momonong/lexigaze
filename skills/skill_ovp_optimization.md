# Skill 7: Optimal Viewing Position (OVP) Optimization

## Context & Objective
To make our Emission Probability $\mathcal{N}(g_t | \mu_i, \Sigma)$ biologically accurate, we must adjust $\mu_i$ (the expected gaze center for a word).
Cognitive science dictates that humans do not look at the exact geometric center of a word. They look at the Optimal Viewing Position (OVP), which is generally shifted slightly to the left of the center, specifically around the center of the first half of the word.

## Required Implementation
Modify our Spatial Probability / Dynamic Cognitive Field generator:
1. Instead of $\mu_x = (box_{left} + box_{right}) / 2$, calculate the OVP.
2. **Rule**: $\mu_x$ should be placed at approximately $35\%$ to $40\%$ of the word's width from the left edge.
   $$\mu_x = box_{left} + (box_{right} - box_{left}) \times 0.35$$
3. Apply a length-based variance $\sigma_x$: Longer words should have a wider horizontal variance (tolerance) than short words.

Refactor the spatial Gaussian emission logic to use these OVP-adjusted centers and length-dependent covariances.