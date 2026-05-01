import numpy as np

class ReadingTransitionMatrix:
    """
    Skill 2: Oculomotor Transition Matrix Builder.
    Builds a transition probability matrix P(w_j | w_i) for a Hidden Markov Model.
    """
    def __init__(self, momentum_forward=1.0, skip_weight=0.5, regression_weight=0.1):
        self.momentum_forward = momentum_forward
        self.skip_weight = skip_weight
        self.regression_weight = regression_weight

    def build_matrix(self, base_cm_array, is_L2_reader=False):
        """
        Constructs an N x N transition matrix.
        base_cm_array: Array of static cognitive mass (higher means more difficult).
        is_L2_reader: If True, increases regression probability.
        """
        n = len(base_cm_array)
        t_matrix = np.zeros((n, n))
        
        reg_w = self.regression_weight * (2.0 if is_L2_reader else 1.0)
        
        for i in range(n):
            # 1. Saccadic Momentum (Forward)
            if i + 1 < n:
                t_matrix[i, i + 1] = 1.0 * self.momentum_forward
            if i + 2 < n:
                # Word Skipping logic: higher prob if i+1 is easy (low CM)
                skip_boost = 1.0 - base_cm_array[i + 1]
                t_matrix[i, i + 2] = 0.5 * self.momentum_forward * (1.0 + self.skip_weight * skip_boost)
            
            # 2. Stay (Refixation)
            # Higher base_cm (difficult word) means higher stay probability
            t_matrix[i, i] = 0.3 * base_cm_array[i]
            
            # 3. Regressions (Backwards)
            for j in range(i):
                # Regression probability decays with distance
                dist_penalty = np.exp(-(i - j))
                # Boost regression if current word is hard
                t_matrix[i, j] = reg_w * base_cm_array[i] * dist_penalty

        # Apply softmax to each row to ensure they sum to 1.0
        row_sums = t_matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        t_matrix /= row_sums
        
        return t_matrix
