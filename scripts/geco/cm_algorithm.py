import torch
import spacy
import numpy as np
import pandas as pd
import math
from transformers import AutoTokenizer, AutoModelForMaskedLM
from scipy.stats import multivariate_normal

class CognitiveMassCalculator:
    """
    Implements Cognitive Mass (CM) extraction using Neuro-Symbolic approaches.
    Supports three modes: Neural Surprisal, Mechanistic Action Score, and Neuro-Symbolic Prior.
    """
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", spacy_model="en_core_web_sm"):
        print(f"⏳ Loading models: {model_name} and {spacy_model}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True, output_attentions=True).to(self.device)
        self.model.eval()
        
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"⚠️ Spacy model {spacy_model} not found. Some features may be limited.")
            self.nlp = None

    def _min_max_normalize(self, values):
        v = np.array(values)
        if v.max() == v.min():
            return np.zeros_like(v)
        return (v - v.min()) / (v.max() - v.min())

    def calculate_mode_1_surprisal(self, text, prompt="Read this as an A2 English learner: "):
        """Mode 1: Role-Conditioned Surprisal (Neural Prior)"""
        full_text = prompt + text
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"][0]
        
        # We only care about the original text tokens, not the prompt
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        text_start_idx = len(prompt_ids) - 1 # Adjusted for [CLS] or similar
        
        surprisals = []
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)
            
            for i in range(text_start_idx, len(input_ids) - 1):
                token_id = input_ids[i].item()
                word_prob = probs[i, token_id].item()
                surprisals.append(-math.log2(word_prob) if word_prob > 0 else 15.0)
        
        return self._min_max_normalize(surprisals)

    def calculate_mode_2_action_score(self, text):
        """Mode 2: Layer-wise Action Score (Mechanistic Prior)"""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"][0]
        
        action_scores = []
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Hidden states is a tuple of (embedding, layer1, layer2, ...)
            # We skip embedding layer
            all_layer_logits = [self.model.cls(hs) for hs in outputs.hidden_states[1:]]
            
            for i in range(1, len(input_ids) - 1):
                token_entropy_sum = 0
                for layer_logits in all_layer_logits:
                    probs = torch.softmax(layer_logits[0, i], dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
                    token_token_entropy_sum = entropy
                    token_entropy_sum += token_token_entropy_sum
                action_scores.append(token_entropy_sum)
                
        return self._min_max_normalize(action_scores)

    def calculate_mode_3_neuro_symbolic(self, text, alpha=0.4, beta=0.4, gamma=0.2):
        """Mode 3: Multi-dimensional Linguistic Gravity (Default)"""
        # 1. Surprisal (Neural)
        s_scores = self.calculate_mode_1_surprisal(text, prompt="")
        
        # 2. Syntax Depth (Symbolic)
        if self.nlp:
            doc = self.nlp(text)
            syntax_depths = []
            for token in doc:
                depth = 0
                temp = token
                while temp.head != temp:
                    depth += 1
                    temp = temp.head
                syntax_depths.append(depth)
            d_scores = self._min_max_normalize(syntax_depths)
        else:
            d_scores = np.zeros_like(s_scores)

        # 3. Age of Acquisition (Lexical Heuristic as proxy if DB missing)
        # Using word length and rare char count as a simple symbolic proxy for AoA
        aoa_proxy = [len(word.text) + (1 if any(c in 'xyzqj' for c in word.text.lower()) else 0) 
                     for word in self.nlp(text)] if self.nlp else np.zeros_like(s_scores)
        a_scores = self._min_max_normalize(aoa_proxy)
        
        # Ensure lengths match (handling tokenization differences)
        min_len = min(len(s_scores), len(d_scores), len(a_scores))
        cm = alpha * s_scores[:min_len] + beta * d_scores[:min_len] + gamma * a_scores[:min_len]
        
        return self._min_max_normalize(cm)

class BayesianGravitySnap:
    """
    Applies Bayes' Theorem to align noisy gaze coordinates to the target word.
    P(t|s) \propto P(s|t) * CM(t)
    """
    def __init__(self, sigma_x=40, sigma_y=30):
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def likelihood(self, gaze_pos, target_pos):
        """P(s|t) - 2D Gaussian Likelihood"""
        rv = multivariate_normal([target_pos[0], target_pos[1]], [[self.sigma_x**2, 0], [0, self.sigma_y**2]])
        return rv.pdf(gaze_pos)

    def snap(self, gaze_pos, words_data):
        """
        gaze_pos: [x, y]
        words_data: List of dicts [{"word": "...", "pos": [x, y], "CM": 0.8}, ...]
        """
        posteriors = []
        for word in words_data:
            prior = word['CM']
            like = self.likelihood(gaze_pos, word['pos'])
            posterior = like * prior
            posteriors.append(posterior)
            
        if not posteriors:
            return None
            
        best_idx = np.argmax(posteriors)
        return words_data[best_idx]

def run_pipeline(text, raw_gaze_df):
    """
    Full pipeline execution.
    """
    calculator = CognitiveMassCalculator()
    cm_scores = calculator.calculate_mode_3_neuro_symbolic(text)
    
    # Process text for alignment
    doc = calculator.nlp(text)
    words_data = []
    # Note: This assumes we have mapping from doc tokens to screen coordinates (true_x, true_y)
    # In a real scenario, we'd get these from the data loader/UI
    for i, token in enumerate(doc):
        if i < len(cm_scores):
            words_data.append({
                "word": token.text,
                "pos": [0, 0], # Placeholder for true_x, true_y
                "CM": cm_scores[i]
            })
            
    # Apply Snap
    snapper = BayesianGravitySnap()
    # ... logic to iterate through gaze points and snap ...
    return words_data

if __name__ == "__main__":
    text = "The quick brown fox jumps over the lazy dog."
    calc = CognitiveMassCalculator()
    cm = calc.calculate_mode_3_neuro_symbolic(text)
    print(f"Cognitive Mass for '{text}':")
    print(cm)
