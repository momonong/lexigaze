import pandas as pd
import numpy as np
import os

def analyze_population(file_path, label):
    print(f"--- Analyzing {label} population ---")
    # Read only necessary columns to save memory
    df = pd.read_csv(file_path, usecols=['PP_NR', 'TRIAL', 'WORD_ID_WITHIN_TRIAL', 'WORD_FIRST_FIXATION_X', 'WORD_FIRST_FIXATION_Y'])
    
    subjects = df['PP_NR'].unique()
    trials = df['TRIAL'].unique()
    
    print(f"Subjects ({len(subjects)}): {subjects}")
    print(f"Total Trials: {len(trials)}")
    
    # Check trial 5 specifically as it's the current benchmark
    trial5 = df[df['TRIAL'] == 5]
    sub_counts = trial5.groupby('PP_NR').size()
    print(f"\nTrial 5 word counts per subject (should be consistent):")
    print(sub_counts.describe())
    
    # Check for coordinate consistency across subjects for Trial 5, Word 1
    word1 = trial5[trial5['WORD_ID_WITHIN_TRIAL'] == 1]
    word1['x'] = pd.to_numeric(word1['WORD_FIRST_FIXATION_X'], errors='coerce')
    word1['y'] = pd.to_numeric(word1['WORD_FIRST_FIXATION_Y'], errors='coerce')
    
    print(f"\nTrial 5, Word 1 coordinates across subjects:")
    print(word1[['PP_NR', 'x', 'y']].dropna().head(10))

if __name__ == "__main__":
    analyze_population('data/geco/L1ReadingData.csv', 'L1')
    analyze_population('data/geco/L2ReadingData.csv', 'L2')
