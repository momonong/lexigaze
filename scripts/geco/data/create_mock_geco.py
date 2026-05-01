import pandas as pd
import numpy as np
import os
import sys

# Add scripts directory to path to import CognitiveMassCalculator
sys.path.append(os.getcwd())
from scripts.geco.core.cm_algorithm import CognitiveMassCalculator

def create_mock_geco():
    """
    Creates a mock GECO dataset for Subject pp01, Trial 5.
    Includes words, bounding boxes, and ground truth gaze coordinates.
    """
    text = "The ubiquitous phenomenon completely bewildered the inexperienced researcher."
    words = text.replace('.', '').split()
    
    # Calculate Base CM
    calculator = CognitiveMassCalculator()
    base_cm = calculator.calculate_mode_3_neuro_symbolic(text)
    
    # Define bounding boxes and ground truth coordinates
    # Assume 1920x1080 screen, text centered at Y=500
    start_x = 200
    word_gap = 20
    char_width = 12
    word_height = 40
    y_center = 500
    
    word_data = []
    current_x = start_x
    
    for i, word in enumerate(words):
        w_len = len(word) * char_width
        box = [current_x, y_center - word_height//2, current_x + w_len, y_center + word_height//2]
        true_x = (box[0] + box[2]) / 2
        true_y = (box[1] + box[3]) / 2
        
        # We might have multiple fixations per word, especially for long/hard words
        # Higher CM -> more fixations
        num_fixations = 1 + int(base_cm[i] * 3)
        
        for _ in range(num_fixations):
            word_data.append({
                'WORD_ID': i + 1,
                'WORD': word,
                'true_x': true_x,
                'true_y': true_y,
                'WORD_TOTAL_READING_TIME': 200 + base_cm[i] * 500, # ms
                'CM': base_cm[i],
                'x_min': box[0],
                'y_min': box[1],
                'x_max': box[2],
                'y_max': box[3]
            })
            
        current_x += w_len + word_gap
        
    df = pd.DataFrame(word_data)
    
    os.makedirs("data/geco", exist_ok=True)
    df.to_csv("data/geco/geco_pp01_trial5_clean.csv", index=False)
    print(f"✅ Mock GECO data created: data/geco/geco_pp01_trial5_clean.csv ({len(df)} samples)")

if __name__ == "__main__":
    create_mock_geco()
