import numpy as np

class AttentionGuidedMatrix:
    """
    Skill 8 & 9: LLM Attention-Guided Oculomotor Transition Matrix (STOCK-T v2).
    Fuses physical oculomotor momentum with cognitive-gated sparse linguistic attention.
    """
    def __init__(self, beta=0.5, mu_saccade=1.0, regression_sensitivity=0.8, top_k_anchors=2):
        """
        beta: Distance penalty for physical saccades.
        mu_saccade: Expected forward jump distance.
        regression_sensitivity: How much CM increases attention reliance (Skill 9).
        top_k_anchors: Number of preceding words to keep in attention (Skill 9).
        """
        self.beta = beta
        self.mu_saccade = mu_saccade
        self.regression_sensitivity = regression_sensitivity
        self.top_k_anchors = top_k_anchors

    def build_matrix(self, num_words, bert_attention_matrix=None, base_cm_array=None):
        """
        Builds the N x N transition matrix with cognitive gating and sparsification.
        num_words: Number of words in the sentence.
        bert_attention_matrix: N x N attention matrix from LLM.
        base_cm_array: Array of Cognitive Mass (required for Skill 9).
        """
        n = num_words
        t_matrix = np.zeros((n, n))
        
        # 1. Physical Prior (P_phys)
        p_phys = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist = abs(j - (i + self.mu_saccade))
                p_phys[i, j] = np.exp(-self.beta * dist)
            # Row-normalize physical prior
            row_sum = p_phys[i].sum()
            if row_sum > 0:
                p_phys[i] /= row_sum

        # 2. Attention Prior (P_attn) with Skill 9 Sparsification
        if bert_attention_matrix is not None:
            p_attn = np.copy(bert_attention_matrix)
            
            for i in range(n):
                # Regression Sparsification: Only keep Top-K anchors for j < i
                regressions = p_attn[i, :i]
                if len(regressions) > self.top_k_anchors:
                    # Find threshold for Top-K
                    top_k_val = np.partition(regressions, -self.top_k_anchors)[-self.top_k_anchors]
                    # Mask elements smaller than threshold
                    p_attn[i, :i][regressions < top_k_val] = 0.0
                
                # Zero out very tiny attention weights globally for that row to reduce noise
                p_attn[i][p_attn[i] < 0.01] = 0.0
                
                # Re-normalize row for attention
                row_sum = p_attn[i].sum()
                if row_sum > 0:
                    p_attn[i] /= row_sum
                else:
                    p_attn[i] = p_phys[i] # Fallback if attention is empty

            # 3. Dynamic Fusion (Skill 9 Adaptive Alpha)
            if base_cm_array is not None:
                # Alpha_i = 1.0 - (CM_i * sensitivity)
                # Higher CM -> Lower alpha -> More reliance on attention regressions
                alphas = 1.0 - (np.array(base_cm_array) * self.regression_sensitivity)
                alphas = np.clip(alphas, 0.3, 1.0)
                
                for i in range(n):
                    t_matrix[i] = alphas[i] * p_phys[i] + (1.0 - alphas[i]) * p_attn[i]
            else:
                # Fallback to Skill 8 global alpha (0.7)
                t_matrix = 0.7 * p_phys + 0.3 * p_attn
        else:
            t_matrix = p_phys

        # Final normalization
        row_sums = t_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        t_matrix /= row_sums
        
        return t_matrix

def print_sample_matrix(matrix, n=10):
    """Prints a sample of the transition matrix for verification."""
    print(f"\n📊 Sample {n}x{n} Transition Matrix (STOCK-T v2 - Skill 9):")
    header = " " * 6 + " ".join([f"W{i:<4}" for i in range(n)])
    print(header)
    for i in range(min(n, len(matrix))):
        row_str = f"W{i:<4} " + " ".join([f"{matrix[i, j]:.3f}" for j in range(min(n, len(matrix)))])
        print(row_str)
