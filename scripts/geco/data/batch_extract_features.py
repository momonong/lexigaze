import os
import pandas as pd
import torch
import math
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import torch
import math
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def extract_features(input_path, lang_label):
    csv_path = input_path.replace(".xlsx", ".csv")
    layout_json_path = f"data/geco/meta/{lang_label.lower()}_consensus_layout.json"
    
    if not os.path.exists(layout_json_path):
        print(f"❌ Consensus layout not found: {layout_json_path}")
        return

    with open(layout_json_path, 'r') as f:
        consensus_layouts = json.load(f)

    if os.path.exists(csv_path):
        print(f"⏳ Reading CSV data from {csv_path}...")
        try:
            df_all = pd.read_csv(csv_path)
        except Exception as e:
            print(f"❌ Failed to read {csv_path}: {e}")
            return
    else:
        print(f"❌ {csv_path} not found. Please convert .xlsx to .csv first.")
        return
            
    subjects = df_all['PP_NR'].unique()
    print(f"👥 Found {len(subjects)} subjects in {lang_label} dataset.")
    
    # Model Setup
    MODEL_NAME = "bert-base-multilingual-cased"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, output_attentions=True).to(device)
    model.eval()

    # Pre-cache surprisals and attention for each trial layout to avoid redundant BERT calls
    # Since layout is same for all subjects in a trial, we only calculate once per trial
    trial_metadata = {}

    for sub in tqdm(subjects, desc=f"Processing {lang_label} Subjects"):
        sub_df = df_all[df_all['PP_NR'] == sub]
        trials = sub_df['TRIAL'].unique()
        
        for trial_id in trials:
            trial_key = str(trial_id)
            if trial_key not in consensus_layouts:
                continue
                
            out_dir = f"data/geco/population/{lang_label}/{sub}/trial_{trial_id}"
            os.makedirs(out_dir, exist_ok=True)
            out_layout = f"{out_dir}/layout.csv"
            out_fixations = f"{out_dir}/fixations.csv"
            out_attn = f"{out_dir}/attention.npy"
            
            if os.path.exists(out_layout) and os.path.exists(out_fixations) and os.path.exists(out_attn):
                continue

            # Get Consensus Layout for this trial
            trial_layout = consensus_layouts[trial_key]
            # Sort words by their ID within trial
            sorted_word_indices = sorted([int(k) for k in trial_layout.keys()])
            
            if trial_key not in trial_metadata:
                # Calculate psycholinguistic features once per trial layout
                sentence_words = [trial_layout[str(i)]['word'].strip() for i in sorted_word_indices]
                
                inputs = tokenizer(sentence_words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512).to(device)
                input_ids = inputs["input_ids"][0]
                word_ids = inputs.word_ids(batch_index=0)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    attention_matrix = outputs.attentions[-1][0].mean(dim=0) 
                    
                    num_words = len(sentence_words)
                    word_attn = np.zeros((num_words, num_words))
                    
                    word_to_first_token = {}
                    for token_idx, word_idx in enumerate(word_ids):
                        if word_idx is not None and word_idx not in word_to_first_token:
                            word_to_first_token[word_idx] = token_idx
                            
                    for i in range(num_words):
                        ti = word_to_first_token.get(i, -1)
                        for j in range(num_words):
                            tj = word_to_first_token.get(j, -1)
                            if ti != -1 and tj != -1:
                                word_attn[i, j] = attention_matrix[ti, tj].item()
                    
                    masked_batch = []
                    valid_indices = []
                    for i in range(num_words):
                        token_index = word_to_first_token.get(i, -1)
                        if token_index != -1 and 0 < token_index < len(input_ids) - 1:
                            masked_ids = input_ids.clone()
                            masked_ids[token_index] = tokenizer.mask_token_id
                            masked_batch.append(masked_ids)
                            valid_indices.append((i, token_index))
                    
                    surprisals = {i: 10.0 for i in range(num_words)}
                    if masked_batch:
                        BATCH_SIZE = 16
                        for b_start in range(0, len(masked_batch), BATCH_SIZE):
                            b_end = min(b_start + BATCH_SIZE, len(masked_batch))
                            batch_input = torch.stack(masked_batch[b_start:b_end]).to(device)
                            batch_outputs = model(batch_input)
                            
                            for idx_in_batch in range(b_end - b_start):
                                original_idx = b_start + idx_in_batch
                                word_idx, token_idx = valid_indices[original_idx]
                                gold_id = input_ids[token_idx].item()
                                logits = batch_outputs.logits[idx_in_batch, token_idx]
                                probs = torch.nn.functional.softmax(logits, dim=-1)
                                word_prob = probs[gold_id].item()
                                surprisals[word_idx] = -math.log2(word_prob) if word_prob > 0 else 15.0

                    layout_results = []
                    for i, word_idx in enumerate(sorted_word_indices):
                        attn_score = word_attn[:, i].sum()
                        surprisal = surprisals[i]
                        cm = surprisal * attn_score
                        word_info = trial_layout[str(word_idx)]
                            
                        layout_results.append({
                            "WORD_ID_WITHIN_TRIAL": word_idx,
                            "WORD": word_info['word'],
                            "true_x": round(word_info['x'], 1),
                            "true_y": round(word_info['y'], 1),
                            "surprisal_score": round(surprisal, 4),
                            "attention_score": round(attn_score, 4),
                            "cognitive_mass": round(cm, 4)
                        })
                    
                    trial_metadata[trial_key] = {
                        "df_layout": pd.DataFrame(layout_results),
                        "word_attn": word_attn
                    }

            # Save layout and attention for this subject (redundant but matches pipeline expectation)
            trial_data = trial_metadata[trial_key]
            trial_data["df_layout"].to_csv(out_layout, index=False)
            np.save(out_attn, trial_data["word_attn"])
                
            # 2. Process Subject's Fixations
            df_sub_trial = sub_df[sub_df['TRIAL'] == trial_id]
            fixation_results = []
            
            # Map valid fixations to the layout indices
            # We use WORD_ID_WITHIN_TRIAL as the link
            for row in df_sub_trial.itertuples():
                if pd.notna(row.WORD_FIRST_FIXATION_X) and pd.notna(row.WORD_FIRST_FIXATION_Y):
                    # Find layout index for this WORD_ID_WITHIN_TRIAL
                    try:
                        layout_idx = sorted_word_indices.index(int(row.WORD_ID_WITHIN_TRIAL))
                        fixation_results.append({
                            "layout_index": layout_idx,
                            "WORD_ID_WITHIN_TRIAL": row.WORD_ID_WITHIN_TRIAL,
                            "fixation_x": row.WORD_FIRST_FIXATION_X,
                            "fixation_y": row.WORD_FIRST_FIXATION_Y,
                            "reading_time": getattr(row, 'WORD_TOTAL_READING_TIME', 0)
                        })
                    except ValueError:
                        continue # Word not in consensus layout
                    
            pd.DataFrame(fixation_results).to_csv(out_fixations, index=False)


if __name__ == "__main__":
    extract_features("data/geco/L1ReadingData.xlsx", "L1")
    extract_features("data/geco/L2ReadingData.xlsx", "L2")

