import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm

def generate_consensus_layout(file_path, output_json):
    print(f"--- Generating Consensus Layout for {file_path} ---")
    # Read only necessary columns
    cols = ['TRIAL', 'WORD_ID_WITHIN_TRIAL', 'WORD', 
            'WORD_FIRST_FIXATION_X', 'WORD_FIRST_FIXATION_Y',
            'WORD_SECOND_FIXATION_X', 'WORD_SECOND_FIXATION_Y',
            'WORD_THIRD_FIXATION_X', 'WORD_THIRD_FIXATION_Y',
            'WORD_LAST_FIXATION_X', 'WORD_LAST_FIXATION_Y']
    
    # We read in chunks to avoid OOM for very large CSVs
    chunk_size = 100000
    all_coords = []
    
    for chunk in tqdm(pd.read_csv(file_path, usecols=cols, chunksize=chunk_size), desc="Reading data"):
        # Melt fixations to get a long list of all gaze points per word
        for i in range(1, 4): # First, Second, Third
            suffix = ['FIRST', 'SECOND', 'THIRD'][i-1]
            tmp = chunk[['TRIAL', 'WORD_ID_WITHIN_TRIAL', 'WORD', f'WORD_{suffix}_FIXATION_X', f'WORD_{suffix}_FIXATION_Y']].copy()
            tmp.columns = ['TRIAL', 'WORD_ID_WITHIN_TRIAL', 'WORD', 'x', 'y']
            all_coords.append(tmp)
        
        # Last fixation
        tmp = chunk[['TRIAL', 'WORD_ID_WITHIN_TRIAL', 'WORD', 'WORD_LAST_FIXATION_X', 'WORD_LAST_FIXATION_Y']].copy()
        tmp.columns = ['TRIAL', 'WORD_ID_WITHIN_TRIAL', 'WORD', 'x', 'y']
        all_coords.append(tmp)
        
    df_coords = pd.concat(all_coords)
    df_coords['x'] = pd.to_numeric(df_coords['x'], errors='coerce')
    df_coords['y'] = pd.to_numeric(df_coords['y'], errors='coerce')
    df_coords = df_coords.dropna(subset=['x', 'y'])
    
    print(f"Total valid gaze points: {len(df_coords)}")
    
    # Group by Trial and Word and take Median
    consensus = df_coords.groupby(['TRIAL', 'WORD_ID_WITHIN_TRIAL']).agg({
        'x': 'median',
        'y': 'median',
        'WORD': 'first'
    }).reset_index()
    
    # Convert to nested dict: trial -> word_idx -> {x, y, word}
    layout_dict = {}
    for row in consensus.itertuples():
        t = str(row.TRIAL)
        w = str(row.WORD_ID_WITHIN_TRIAL)
        if t not in layout_dict:
            layout_dict[t] = {}
        layout_dict[t][w] = {
            'x': float(row.x),
            'y': float(row.y),
            'word': str(row.WORD)
        }
    
    with open(output_json, 'w') as f:
        json.dump(layout_dict, f)
    
    print(f"✅ Consensus layout saved to {output_json}")

if __name__ == "__main__":
    os.makedirs('data/geco/meta', exist_ok=True)
    generate_consensus_layout('data/geco/L1ReadingData.csv', 'data/geco/meta/l1_consensus_layout.json')
    generate_consensus_layout('data/geco/L2ReadingData.csv', 'data/geco/meta/l2_consensus_layout.json')
