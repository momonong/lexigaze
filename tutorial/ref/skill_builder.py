import os
import argparse
import json
from datetime import datetime
from diagnostic_agent import get_context, call_llm

SKILL_TEMPLATE = """# Skill: 認知科學與眼動軌跡分析 (Cognitive Gaze Analysis)

## 📌 Metadata
- **知識領域**：心理語言學、二語習得 (SLA)、眼動科學
- **最後更新時間**：{date}
- **適用 Agent 類型**：閱讀診斷專家 / 學習輔導 Copilot

## 👩‍🔬 學生小組討論區 (Student Review Section)
【小組任務】：以下 `Core Parameters` 是 AI 從高階眼動儀的實驗室文獻中萃取出的標準參數。
但在我們使用低成本 Webcam (WebGazer) 的環境中，雜訊較大。
👉 **請討論：我們是否應該手動放寬容錯率（例如把 Fixation 的時間閾值從 200ms 拉長到 300ms），以避免雜訊干擾？**
👉 **請直接修改下方 JSON 內的數值，這將決定我們最終診斷 Agent 的敏感度！**

## ⚙️ 核心參數設定 (Core Parameters)
```json
{parameters}
```

## 📖 理論基礎 (Overview & Theory)
{overview}

## 👁️ 軌跡特徵解讀 (Gaze Features)
{features}

## 💡 教學介入策略 (Pedagogical Strategies)
{strategies}
"""

QUESTIONS = {
    "parameters": "請從文獻中提取眼動追蹤的核心參數（如：注視時間 Fixation Duration 的毫秒閾值、掃視 Saccade 速度、回視 Regression 發生的條件、Surprisal 驚奇度的關聯）。請【只輸出一段 JSON 格式的內容】，包含鍵值對，不要有任何多餘文字。",
    "overview": "簡述眼動軌跡（如 Fixation 和 Regression）如何反映閱讀者的認知負荷與語意理解困難？限 200 字以內。",
    "features": "詳細列出 3-5 種關鍵的眼動軌跡特徵（例如長時間注視生難字、頻繁回視上一句），並說明這些特徵在心理語言學上代表什麼樣的學習者掙扎狀態？",
    "strategies": "當我們透過眼動偵測到學習者在特定單字或複雜句法上卡關時，文獻建議可以採取哪些具體的教學介入策略（Pedagogical Strategies）？"
}

def ask_rag_for_skill(query, model_name=None):
    context, sources = get_context(query, category="Academic", top_k=5)
    if not context.strip():
        context = "無相關文獻，請基於認知科學專業知識回答。"
    
    SYSTEM_PROMPT = "你是一位頂尖的「認知科學與心理語言學」研究員。請根據文獻回答問題，幫助建立 Agent Skill。"
    USER_PROMPT = f"【文獻】\n{context}\n\n【問題】\n{query}"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT}
    ]
    answer = call_llm(messages, model_name=model_name)
    return answer, sources

def build_skill(output_path="skill.md", model="gemini-2.5-flash", append=False):
    print(f"🚀 開始自動生成/更新 {output_path}...")
    responses = {}
    all_sources = set()

    for key, q in QUESTIONS.items():
        print(f"  > 正在處理: {key}...")
        answer, sources = ask_rag_for_skill(q, model_name=model)
        
        # 清理 JSON 輸出，避免 markdown code block 格式殘留
        if key == "parameters":
            answer = answer.replace("```json", "").replace("```", "").strip()
            # 確保大括號存在
            if not answer.startswith("{"):
                answer = "{\n" + answer
            if not answer.endswith("}"):
                answer = answer + "\n}"
                
        responses[key] = answer
        all_sources.update(sources)

    print("📝 正在組裝 Skill 文件...")
    sources_list = "\n".join([f"- {s}" for s in sorted(list(all_sources))])
    
    final_content = SKILL_TEMPLATE.format(
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        parameters=responses["parameters"],
        overview=responses["overview"],
        features=responses["features"],
        strategies=responses["strategies"],
    )

    if sources_list:
        final_content += f"\n## 📚 引用文獻\n{sources_list}\n"

    mode = "a" if append and os.path.exists(output_path) else "w"
    
    # 如果是 append mode，加上分隔線與時間戳記
    if mode == "a":
        final_content = f"\n\n---\n\n# 🔄 Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n" + final_content

    with open(output_path, mode, encoding="utf-8") as f:
        f.write(final_content)

    print(f"✅ 成功！Skill 文件已{'附加' if mode=='a' else '覆寫'}至 {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="skill.md")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--append", action="store_true", help="是否以附加模式寫入檔案")
    args = parser.parse_args()
    build_skill(args.output, args.model, args.append)
