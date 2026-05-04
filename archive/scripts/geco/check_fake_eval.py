
import pandas as pd
import numpy as np

def check_fake_eval():
    L1_DATA_CSV = "data/geco/geco_l1_pp01_cognitive_mass.csv"
    if not pd.io.common.file_exists(L1_DATA_CSV):
        print("Data file not found")
        return
        
    df = pd.read_csv(L1_DATA_CSV)
    print(f"Total rows: {len(df)}")
    
    word_boxes = [[row['true_x']-20, row['true_y']-15, row['true_x']+20, row['true_y']+15] for _, row in df.iterrows()]
    targets = np.arange(len(df))
    
    # Check if word boxes are repeated
    unique_boxes = []
    for box in word_boxes:
        if box not in unique_boxes:
            unique_boxes.append(box)
            
    print(f"Number of word boxes given to decoder: {len(word_boxes)}")
    print(f"Number of unique word boxes in trial: {len(unique_boxes)}")
    
    if len(word_boxes) > len(unique_boxes):
        print("⚠️ PROBLEM: The decoder is being given the ground truth sequence of boxes as candidates!")
        print("It just needs to stay on the diagonal to get 100% accuracy.")
    else:
        print("No obvious diagonal bias in box count, but checking if boxes are ordered by visitation...")

if __name__ == "__main__":
    check_fake_eval()
