import os
import argparse
import pandas as pd
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# --- 資料庫設定 ---
DB_PARAMS = {
    "dbname": "ragdb",
    "user": "raguser",
    "password": "ragpassword",
    "host": "localhost",
    "port": "5432"
}

# --- 診斷 Agent Prompt 模板 ---
DIAGNOSTIC_SYSTEM_PROMPT = """你是一位專精於「心理語言學 (Psycholinguistics)」與「二語習得 (SLA)」的診斷專家。
你的任務是根據學生的「真實眼動軌跡特徵」與「閱讀文本」，並結合檢索到的【認知科學文獻】，來診斷學生的閱讀理解狀態。

請注意：
1. **理論支持**：你的診斷必須基於眼動科學（如：長時間 Fixation 代表認知負荷高，Regression (回視) 代表句法重構或理解困難）。
2. **數據導向**：請具體引用傳入的眼動數據特徵。
3. **同理心與建設性**：請用溫和、專業的語氣給出「教學建議 (Pedagogical Advice)」，幫助學習者克服困難。
4. **誠實告知**：若文獻中缺乏相關支持，請根據你的 SLA 專業知識進行合理推論，但必須註明是推論。

回答格式：
- **👁️ 軌跡特徵分析**：分析傳入的眼動數據現象。
- **🧠 學習者掙扎點 (Struggle Points)**：明確指出學生可能在哪個單字或句法結構卡關。
- **💡 教學建議 (Pedagogical Advice)**：給出具體且可操作的輔導策略。
- **📚 引用文獻**：附上支持你診斷的文獻片段來源。"""

DIAGNOSTIC_USER_TEMPLATE = """【閱讀文本】：
{reading_text}

【學生眼動軌跡摘要】：
{gaze_summary}

【檢索到的認知科學文獻】：
{context}

請根據上述資料提供深度診斷報告："""

# --- 檢索核心邏輯 ---
def get_context(query, category="Academic", top_k=3):
    """從向量資料庫檢索最相關的學術片段"""
    from sentence_transformers import SentenceTransformer
    try:
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        query_vector = model.encode(query).tolist()
        
        conn = psycopg2.connect(**DB_PARAMS)
        register_vector(conn)
        cur = conn.cursor()
        
        sql = """
            SELECT doc_name, content, 1 - (embedding <=> %s::vector) AS similarity
            FROM documents
            WHERE category = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        cur.execute(sql, (query_vector, category, query_vector, top_k))
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        context_str = ""
        sources = []
        for i, (doc_name, content, sim) in enumerate(results):
            context_str += f"--- 文獻 {i+1}: {doc_name} ---\n{content}\n\n"
            sources.append(doc_name)
            
        return context_str, sources
    except Exception as e:
        print(f"⚠️ 資料庫連線或檢索失敗: {e}")
        return "", []

# --- LLM 呼叫核心邏輯 (雙路徑 Fallback) ---
def call_llm(messages, model_name=None):
    proxy_key = os.getenv("LITELLM_API_KEY")
    proxy_url = os.getenv("LITELLM_BASE_URL")
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if not model_name:
        model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")

    if proxy_key and proxy_url:
        try:
            from litellm import completion
            print(f"🚀 嘗試透過 Proxy 呼叫: {model_name}")
            response = completion(
                model=f"openai/{model_name}" if not model_name.startswith("openai/") else model_name, 
                messages=messages,
                api_base=proxy_url,
                api_key=proxy_key,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ Proxy 呼叫失敗，嘗試備援路徑... (Error: {str(e)[:50]})")

    if gemini_key:
        try:
            clean_model_name = model_name.replace("gemini/", "") if model_name.startswith("gemini/") else model_name
            if clean_model_name == "gemini-1.5-flash":
                clean_model_name = "gemini-2.5-flash"
                
            print(f"✅ 使用 Gemini 原生 SDK 備援路徑: {clean_model_name}")
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=gemini_key)
            system_instruction = next((m["content"] for m in messages if m["role"] == "system"), None)
            user_message = next((m["content"] for m in messages if m["role"] == "user"), "")
            
            config = types.GenerateContentConfig(
                temperature=0.3,
                system_instruction=system_instruction,
            )
            
            response = client.models.generate_content(
                model=clean_model_name,
                contents=user_message,
                config=config
            )
            return response.text
        except ImportError:
            return "❌ 錯誤：未安裝 google-genai SDK。"
        except Exception as e:
            return f"❌ 所有 LLM 路徑皆失敗: {str(e)}"
    return "❌ 錯誤：未設定 API_KEY"

# --- 診斷 Agent 核心工具 ---
def diagnose_learner(csv_path, reading_text, model_name=None):
    """
    作為一個 Agent Tool，接收眼動特徵與文本，輸出診斷建議。
    """
    print(f"📊 正在分析眼動資料: {csv_path}")
    if not os.path.exists(csv_path):
        return "找不到指定的 CSV 眼動資料檔。", []
        
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return "眼動資料為空。", []

    # --- 萃取眼動特徵摘要 (模擬 Skills 的運作) ---
    total_time_ms = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]) if len(df) > 1 else 0
    data_points = len(df)
    
    # 假設目標單字在 (850, 500) 附近，簡單計算在該區域的資料點數
    target_x, target_y = 850, 500
    radius = 150
    df['dist_to_target'] = ((df['x_px'] - target_x)**2 + (df['y_px'] - target_y)**2)**0.5
    points_near_target = len(df[df['dist_to_target'] < radius])
    
    gaze_summary = f"- 總閱讀時間：{total_time_ms} 毫秒\n"
    gaze_summary += f"- 總注視資料點：{data_points} 點\n"
    gaze_summary += f"- 針對生難字區域 (X:{target_x}, Y:{target_y}) 附近的資料點數量：{points_near_target} 點\n"
    if points_near_target > (data_points * 0.3):
         gaze_summary += "- ⚠️ 系統偵測到學習者在目標生難字周圍產生了高密度的軌跡 (可能為 Regression 或過長 Fixation)。\n"

    # 使用診斷關鍵字進行文獻檢索
    search_query = "eye tracking fixation duration reading difficulty cognitive load surprisal regression"
    print("🔍 正在從認知科學知識庫檢索相關文獻...")
    context, sources = get_context(search_query, top_k=3)
    
    if not context.strip():
        print("⚠️ 知識庫中無相關文獻，將僅依賴 LLM 的內建心理語言學知識。")

    messages = [
        {"role": "system", "content": DIAGNOSTIC_SYSTEM_PROMPT},
        {"role": "user", "content": DIAGNOSTIC_USER_TEMPLATE.format(
            reading_text=reading_text, 
            gaze_summary=gaze_summary, 
            context=context
        )}
    ]
    
    print("🧠 診斷專家 Agent 正在生成報告...")
    answer = call_llm(messages, model_name=model_name)
    return answer, sources

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="心理語言學診斷 Agent")
    parser.add_argument("--csv", type=str, default="../data/calibrated.csv", help="校準後的眼動軌跡 CSV 路徑")
    parser.add_argument("--model", type=str, help="指定的模型名稱")
    args = parser.parse_args()

    sample_text = (
        "The ubiquitous phenomenon completely bewildered the inexperienced researcher.\n"
        "Despite rigorous analysis, the underlying mechanisms remained enigmatic, defying conventional explanation."
    )

    # 確保以正確的相對路徑讀取 (假設從 tutorial/ref 執行)
    csv_path = args.csv
    if not os.path.exists(csv_path) and os.path.exists(os.path.join("..", "data", "calibrated.csv")):
        csv_path = os.path.join("..", "data", "calibrated.csv")

    answer, sources = diagnose_learner(csv_path, sample_text, model_name=args.model)
    print(f"\n================ 🩺 診斷報告 ================\n\n{answer}")
    if sources:
        print("\n📚 引用文獻:", ", ".join(list(set(sources))))
