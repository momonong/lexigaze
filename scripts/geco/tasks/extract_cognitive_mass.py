import os
from dotenv import load_dotenv

# 1. 載入 .env 檔案中的環境變數
load_dotenv()

# 2. (可選) 印出來確認一下有沒有成功讀到
print(f"📦 Hugging Face 模型存放路徑: {os.getenv('HF_HOME')}")

import pandas as pd
import torch
import math
from transformers import AutoTokenizer, AutoModelForMaskedLM

# 1. 載入我們清洗好的乾淨 ESL 閱讀資料
input_file = "data\geco\geco_pp01_trial5_clean.csv"
output_file = "data\geco\geco_pp01_cognitive_mass.csv"

print("⏳ 載入資料集...")
df = pd.read_csv(input_file)

# 把所有的單字串成一個完整的句子，供 BERT 閱讀上下文
sentence_words = df['WORD'].tolist()
full_sentence = " ".join(sentence_words)

# 2. 載入模型 (RTX 5090 準備發威！)
# 雖然 bert-tiny 很小，但使用 GPU 還是能顯著加速
MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"
print(f"📦 載入神經網路模型: {MODEL_NAME}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用運算設備: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, output_attentions=True).to(device)
model.eval()

# 3. 核心運算：萃取 Surprisal 與 Attention
print("🧠 正在計算神經符號特徵 (這可能需要幾秒鐘)...")

inputs = tokenizer(full_sentence, return_tensors="pt").to(device)
input_ids = inputs["input_ids"][0]
tokens = tokenizer.convert_ids_to_tokens(input_ids)

results = []

with torch.no_grad():
    outputs = model(**inputs)
    # 提取最後一層的 Attention Matrix (Layer -1), 取所有 Head 的平均值
    # 形狀: [batch_size, num_heads, seq_length, seq_length]
    attention_matrix = outputs.attentions[-1][0].mean(dim=0) 

    # 針對每一個被 GECO 記錄下來的單字計算特徵
    for idx, row in df.iterrows():
        target_word = str(row['WORD']).lower()
        
        # 尋找該單字在 BERT token 列表中的對應位置 (簡單匹配)
        token_index = -1
        for i, token in enumerate(tokens):
            if token_index == -1 and token.replace("##", "") == target_word:
                token_index = i
                break
                
        if token_index != -1 and token_index > 0 and token_index < len(input_ids) - 1:
            gold_id = input_ids[token_index].item()
            
            # A. 計算 Surprisal (局部難度)
            masked_ids = input_ids.clone()
            masked_ids[token_index] = tokenizer.mask_token_id
            
            mask_outputs = model(masked_ids.unsqueeze(0))
            logits = mask_outputs.logits[0, token_index]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            word_prob = probs[gold_id].item()
            surprisal = -math.log2(word_prob) if word_prob > 0 else 15.0
            
            # B. 計算 Attention Centrality (全局重要性)
            # 看看句子中其他字 (j) 有多少注意力放在這個目標字 (token_index) 上
            attention_score = attention_matrix[:, token_index].sum().item()
            
            # C. 融合為 Cognitive Mass 認知質量
            # 論文核心公式: Mass = Surprisal * Attention
            cognitive_mass = surprisal * attention_score
            
            results.append({
                "WORD_ID": row["WORD_ID"],
                "WORD": row["WORD"],
                "true_x": row["true_x"],
                "true_y": row["true_y"],
                "surprisal_score": round(surprisal, 4),
                "attention_score": round(attention_score, 4),
                "cognitive_mass": round(cognitive_mass, 4)
            })
        else:
            # 如果 Tokenizer 拆分導致對應失敗，給予預設值
             results.append({
                "WORD_ID": row["WORD_ID"],
                "WORD": row["WORD"],
                "true_x": row["true_x"],
                "true_y": row["true_y"],
                "surprisal_score": 5.0,
                "attention_score": 0.5,
                "cognitive_mass": 2.5
            })

# 4. 儲存結果，準備下班！
df_results = pd.DataFrame(results)
df_results.to_csv(output_file, index=False)

print(f"✅ 運算完成！已將包含『認知質量』的最終特徵檔儲存為 {output_file}")
print("\n預覽前 3 筆最高質量的單字 (這些字會產生最強的引力)：")
print(df_results.sort_values(by="cognitive_mass", ascending=False).head(3))