import os
import pandas as pd
import torch
import math
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

def extract_features(input_path, lang_label, subjects, num_trials=10):
    print(f"⏳ Processing {lang_label} data from {input_path}...")
    df_all = pd.read_excel(input_path)
    
    # Model Setup
    MODEL_NAME = "bert-base-multilingual-cased"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, output_attentions=True).to(device)
    model.eval()

    for sub in subjects:
        for trial_id in range(1, num_trials + 1):
            out_dir = f"data/geco/population/{lang_label}/{sub}/trial_{trial_id}"
            os.makedirs(out_dir, exist_ok=True)
            out_csv = f"{out_dir}/features.csv"
            out_attn = f"{out_dir}/attention.npy"
            
            if os.path.exists(out_csv):
                continue
                
            df_filtered = df_all[(df_all['PP_NR'] == sub) & (df_all['TRIAL'] == trial_id)].copy()
            if len(df_filtered) == 0:
                print(f"⚠️ Skip: {sub} Trial {trial_id} not found.")
                continue
                
            # Clean data
            df_filtered = df_filtered[df_filtered['WORD_FIRST_FIXATION_X'] != '.'].copy()
            if len(df_filtered) < 10: # Skip too short trials
                continue
                
            df_filtered['true_x'] = pd.to_numeric(df_filtered['WORD_FIRST_FIXATION_X'])
            df_filtered['true_y'] = pd.to_numeric(df_filtered['WORD_FIRST_FIXATION_Y'])
            
            # CM Extraction
            sentence_words = df_filtered['WORD'].tolist()
            full_sentence = " ".join([str(w) for w in sentence_words])
            
            inputs = tokenizer(full_sentence, return_tensors="pt", truncation=True, max_length=512).to(device)
            input_ids = inputs["input_ids"][0]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            
            with torch.no_grad():
                outputs = model(**inputs)
                attention_matrix = outputs.attentions[-1][0].mean(dim=0) 
                
                word_token_indices = []
                for _, row in df_filtered.iterrows():
                    target_word = str(row['WORD']).strip().lower()
                    token_index = -1
                    for i, token in enumerate(tokens):
                        if token.replace("##", "").lower() == target_word:
                            token_index = i
                            break
                    word_token_indices.append(token_index)
                
                num_words = len(df_filtered)
                word_attn = np.zeros((num_words, num_words))
                for i in range(num_words):
                    ti = word_token_indices[i]
                    for j in range(num_words):
                        tj = word_token_indices[j]
                        if ti != -1 and tj != -1:
                            word_attn[i, j] = attention_matrix[ti, tj].item()
                            
                results = []
                for i, row in df_filtered.iterrows():
                    idx_in_trial = i # This is not correct for results list
                    # Find relative index in the filtered dataframe
                    rel_idx = list(df_filtered.index).index(i)
                    
                    token_index = word_token_indices[rel_idx]
                    if token_index != -1 and 0 < token_index < len(input_ids) - 1:
                        gold_id = input_ids[token_index].item()
                        masked_ids = input_ids.clone()
                        masked_ids[token_index] = tokenizer.mask_token_id
                        mask_outputs = model(masked_ids.unsqueeze(0))
                        logits = mask_outputs.logits[0, token_index]
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        word_prob = probs[gold_id].item()
                        surprisal = -math.log2(word_prob) if word_prob > 0 else 15.0
                        attn_score = word_attn[:, rel_idx].sum()
                        cm = surprisal * attn_score
                        results.append({
                            "WORD_ID": row["WORD_ID"], "WORD": row["WORD"],
                            "true_x": row["true_x"], "true_y": row["true_y"],
                            "surprisal_score": round(surprisal, 4),
                            "attention_score": round(attn_score, 4),
                            "cognitive_mass": round(cm, 4),
                            "WORD_TOTAL_READING_TIME": row["WORD_TOTAL_READING_TIME"]
                        })
                    else:
                        results.append({
                            "WORD_ID": row["WORD_ID"], "WORD": row["WORD"],
                            "true_x": row["true_x"], "true_y": row["true_y"],
                            "surprisal_score": 10.0, "attention_score": 0.5,
                            "cognitive_mass": 5.0, "WORD_TOTAL_READING_TIME": row["WORD_TOTAL_READING_TIME"]
                        })
                
                pd.DataFrame(results).to_csv(out_csv, index=False)
                np.save(out_attn, word_attn)
                print(f"✅ Saved {sub} Trial {trial_id} ({lang_label})")

if __name__ == "__main__":
    subjects = ['pp01', 'pp02', 'pp03', 'pp04', 'pp05']
    extract_features("data/geco/L1ReadingData.xlsx", "L1", subjects)
    extract_features("data/geco/L2ReadingData.xlsx", "L2", subjects)
