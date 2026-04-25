import os
from dotenv import load_dotenv
load_dotenv()

# 1. 強制設定 Hugging Face 模型下載到 D 槽
# 你可以自己更改後面的資料夾名稱
os.environ["HF_HOME"] = r"D:\hf_models" 

# 2. 關閉 Windows 環境下的 symlink 警告 (選用，讓終端機乾淨一點)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

    
import math
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForMaskedLM
import torch.nn.functional as F

# 檢查 GPU 狀態 (教學重點：讓學生了解硬體加速)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ GPU 已啟用: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️ 未偵測到 GPU，將使用 CPU 執行 (bert-tiny 在 CPU 上依然極快)。")
    print("👉 提示：你可以點擊上方選單「執行階段 > 變更執行階段類型 > T4 GPU」。")

print("🚀 啟動 IntelligentGaze 神經符號融合引擎 (Neuro-Symbolic Engine)...")

# 設定資料目錄
DATA_DIR = r"tutorial\data"
os.makedirs(DATA_DIR, exist_ok=True)

# ==========================================
# 階段一：符號先驗萃取 (Symbolic Prior Extraction)
# ==========================================
print("\n📦 [1/3] 正在載入語言模型 (bert-tiny)...")
tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = BertForMaskedLM.from_pretrained("prajjwal1/bert-tiny")

text = "The ubiquitous phenomenon completely bewildered the inexperienced researcher."
target_word = "phenomenon"
word_x, word_y = 450, 320  # 假設單字在螢幕上的座標

def calculate_surprisal(sentence, word):
    masked = sentence.replace(word, tokenizer.mask_token)
    inputs = tokenizer(masked, return_tensors="pt")
    mask_idx = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    
    with torch.no_grad():
        logits = model(**inputs).logits
        
    probs = F.softmax(logits[0, mask_idx[0], :], dim=-1)
    target_id = tokenizer.convert_tokens_to_ids(word.lower())
    prob = probs[target_id].item()
    
    return -math.log(prob) if prob > 0 else 20.0

print(f"📖 分析文本: '{text}'")
surprisal = calculate_surprisal(text, target_word)
alpha_weight = min(surprisal / 15.0, 0.95)

# 輸出知識庫 JSON
knowledge_base = {
    target_word: {"x": word_x, "y": word_y, "surprisal": round(surprisal, 2), "alpha": round(alpha_weight, 2)}
}
kb_path = os.path.join(DATA_DIR, "cognitive_weights.json")
with open(kb_path, "w") as f:
    json.dump(knowledge_base, f, indent=4)
print(f"✅ 驚奇度計算完成: {target_word} (Alpha={alpha_weight:.2f}) -> 已儲存至 {kb_path}")


# ==========================================
# 階段二：神經感知載入 (Neural Perception Loading)
# ==========================================
print("\n📊 [2/3] 正在載入硬體眼動數據...")
csv_path = os.path.join(DATA_DIR, "my_eyes.csv")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f"✅ 成功載入真實數據: {len(df)} 筆。")
else:
    print(f"⚠️ 找不到 {csv_path}，啟用防護機制：自動生成模擬雜訊數據。")
    np.random.seed(42)
    df = pd.DataFrame({
        'x_px': np.random.normal(loc=430, scale=80, size=50),
        'y_px': np.random.normal(loc=350, scale=60, size=50)
    })

# 建立 Baseline (移動平均)
df['baseline_x'] = df['x_px'].rolling(window=5, min_periods=1).mean()
df['baseline_y'] = df['y_px'].rolling(window=5, min_periods=1).mean()


# ==========================================
# 階段三：神經符號融合 (Neuro-Symbolic Calibration)
# ==========================================
print("\n🧲 [3/3] 執行引力吸附演算法...")

def apply_gravity(row):
    # 以 Baseline 為基礎進行校準，避免高頻雜訊干擾引力判斷
    px, py = row['baseline_x'], row['baseline_y']
    dist = np.sqrt((px - word_x)**2 + (py - word_y)**2)
    
    if dist < 150: # 啟動半徑
        calib_x = px + (word_x - px) * alpha_weight
        calib_y = py + (word_y - py) * alpha_weight
        return pd.Series([calib_x, calib_y])
    return pd.Series([px, py])

df[['vibe_x', 'vibe_y']] = df.apply(apply_gravity, axis=1)

# 輸出最終乾淨的資料
output_path = os.path.join(DATA_DIR, "calibrated_eyes.csv")
df.to_csv(output_path, index=False)
print(f"✅ 校準完成！融合結果已匯出至 {output_path}")


# ==========================================
# 視覺化展示 (Dashboard)
# ==========================================
plt.figure(figsize=(10, 6))

# 1. 原始雜訊
plt.scatter(df['x_px'], df['y_px'], color='lightcoral', alpha=0.3, label='1. Raw Webcam Noise')
# 2. 對照組
plt.plot(df['baseline_x'], df['baseline_y'], color='orange', linewidth=1.5, linestyle='--', label='2. Baseline (Moving Average)')
# 3. 實驗組 (神經符號融合)
plt.scatter(df['vibe_x'], df['vibe_y'], color='darkblue', alpha=0.8, s=60, label='3. Neuro-Symbolic Gaze')

# 標示引力目標與啟動半徑
plt.scatter(word_x, word_y, color='green', marker='*', s=300, edgecolor='black', zorder=5)
plt.text(word_x + 15, word_y - 15, f"{target_word}\n(Alpha: {alpha_weight:.2f})", fontsize=11, fontweight='bold', color='green')
circle = plt.Circle((word_x, word_y), 150, color='green', fill=False, linestyle=':', alpha=0.5)
plt.gca().add_patch(circle)

plt.title('Neuro-Symbolic Engine: Surprisal-Driven Calibration', fontsize=16)
plt.xlabel('Screen X')
plt.ylabel('Screen Y')
plt.gca().invert_yaxis()
plt.legend(loc='lower left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("tutorial/figures/dashboard.png")
plt.show()