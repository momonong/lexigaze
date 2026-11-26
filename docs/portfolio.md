# LexiGaze 多模態閱讀分析系統

## 專案概覽

LexiGaze 是一個把眼動與語言特徵對齊到單字的閱讀研究原型。研究者可比較讀者在哪些字停留或回看，以及這些位置有哪些文字特徵，再檢視兩類訊號是否指向同一段閱讀難點。見 [專案說明](../README.md)。

## 使用流程

研究介面先擷取文件中每個詞的畫面座標，接著做九點視線校正。閱讀時，系統把相機預測的視線對到詞框，並可顯示文字特徵或融合結果的熱圖。見 [整合示範流程](demo_integration_guide.md)。

獨立的本機閱讀工作區提供另一條流程。讀者閱讀後鎖定客觀作答，再填寫信心與複習偏好，最後回顧原文、答案及解釋。完成的紀錄可以下載成 JSON，在瀏覽器重新開啟，由讀者自行保存。這條流程關閉相機。見 [本機閱讀與作答流程](https://github.com/momonong/lexigaze/blob/e6d0783e01357fb45eefb3a6e85083efb2c55fc7/docs/CHI/2026-09-07-reading-learner-profile-core-v1.md)。

## 系統架構

Flask 介面串接視線、語言分析與融合模組。視線路徑用 MediaPipe 擷取臉部資訊、UniGaze 預測畫面座標，再以個別校正減少偏移。語言路徑以 GPT-2 或 BERT 等工具計算詞彙與句法特徵，融合模組依同一份詞框佈局整合注視停留和文字特徵，產生實驗性的逐詞閱讀難度分數。見 [系統架構](../ARCHITECTURE.md)。

## 實作與貢獻

團隊專案實作了詞框對齊、視線校正與多種融合方法，也以序列解碼和校正模型研究視線落在相鄰行的問題。本機閱讀紀錄另保存題庫來源、時間、客觀選項、自評缺漏與主動選擇的複習項目，讓回顧畫面能對應當時的原文和作答。見 [眼動實驗結果](NeurIPS/RESULT.md)與[紀錄格式與操作](https://github.com/momonong/lexigaze/blob/e6d0783e01357fb45eefb3a6e85083efb2c55fc7/docs/CHI/2026-09-07-reading-learner-profile-core-v1.md)。

## 成果與限制

可攜閱讀紀錄整合版本通過 36 項軟體測試，涵蓋作答流程及紀錄匯出後重新開啟。這一版的實際瀏覽器檔案操作與手機驗收仍待進行。眼動校正結果來自指定資料集和實驗條件，題庫效度、個人能力評量與學習成效仍待研究。見 [主分支整合驗證](https://github.com/momonong/lexigaze/blob/e6d0783e01357fb45eefb3a6e85083efb2c55fc7/docs/CHI/2026-09-07-reading-learner-profile-core-v1.md#2026-10-26-main-%E5%9F%BA%E5%BA%95%E6%95%B4%E5%90%88%E9%A9%97%E8%AD%89)與[眼動實驗結果](NeurIPS/RESULT.md)。
