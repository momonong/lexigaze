# 閱讀、作答與 learner profile：核心流程開發版

日期：2026-09-07。狀態：單次核心流程已實作，軟體與瀏覽器操作驗證通過。

## 本輪目標與範圍

Owner 在本輪選擇「把閱讀、作答與 learner profile 串成完整流程」。本輪實作一個可操作的本機開發版本：

> 閱讀 → 明確完成閱讀 → 客觀作答 → 信心與複習偏好 → 本次學習檔案 → 回看自行選擇的複習項目。

本次 profile 是帶來源與時間的描述性學習紀錄；不擬合能力、不產生認知診斷、不宣稱 profile 效度或學習成效。
它先完成一次使用期間的資料與互動連結。跨次保存、獨立能力評量、延後測驗及適性推薦是後續工作。

## 現況核對與設計決策

- 最新 checkout：`experiment/chi-gaze-reliability-v1` 的 `93d845d` 加上既有未提交工作；本輪從它建立 `codex/reading-learner-profile-flow`。
- 既有 `/objective-profile` 已有兩篇短文、十題與分項結果，但在閱讀前會傳送所有問題，沒有 server-held 閱讀與答案階段。
- 既有 participant profile 是研究流程的背景記錄，不能直接當成會更新的 learner model。
- 新增獨立 `/learner-workspace`，保留 v1 walkthrough、既有研究流程、原題庫與不可變證據。
- 原題庫僅以唯讀方式重用為**未驗證的示例內容**。新流程是產品開發預覽，不取代正式測量或介入 protocol。

## 使用流程

1. 首頁說明本次內容、暫存範圍與相機停用。
2. 閱讀兩篇文章及兩段新語境，各段先只傳送閱讀內容。
3. 按下完成閱讀後，由 server 固定該段開始／結束時間，再提供本段問題。
4. 每題先鎖定客觀答案，再取得信心與複習偏好。`I do not know` 不要求信心；跳過自評與不知道分別記錄。
5. 十題完成後呈現語境字義、文章理解、新語境作答及完整逐題記錄。
6. 以使用者自己選擇的複習項目篩選回顧；回顧保留對應原文、選擇、答案與解釋。

答案、計分類型與解釋只在全部作答完成後顯示，避免先給回饋影響後續新語境作答。新語境題仍是同次使用，不能稱為延後保留。
閱讀時間是 server 記錄的經過時間，可能包含閒置；不推導有效注視時間、WPM 或閱讀能力。

## Profile 的資料意義

| 資料 | 用途 | 邊界 |
|---|---|---|
| 分項作答與各題結果 | 回看本次哪些內容答對／不知道／未答對 | 不換算整體能力分數 |
| 鎖定答案後的信心 | 保留對自己答案的判斷 | 與 correctness 分開；不推論穩定認知特質 |
| 複習偏好 | 直接控制使用者選擇的回顧清單 | 不等同答錯、學習需求或介入效益 |
| 閱讀、答案、觀察時間與來源 | 讓後續模型能辨識證據先後與所用內容 | 後於答案的資料不回流成同次介入前預測 |
| 尚未量測的項目 | 明示延後保留、gaze、認知能力尚無資料 | 不生成模擬或推測數值補空格 |

## 軟體邊界

- 專用 loopback authority，預設 `127.0.0.1:8097`；route/method/origin 檢查與 no-store headers。
- Session token 僅在瀏覽器記憶體中持有，server 以有限容量與 30 分鐘 TTL 暫存。沒有帳號、cookie、瀏覽器持久儲存或資料庫。
- 重新開始會刪除 session。離開頁面嘗試刪除；瀏覽器未完成刪除請求時，session 建立 30 分鐘後無法存取，並於下一次 store 操作或 process 結束時清除，不保證關頁或到期瞬間即從記憶體移除。
- 重複／過期 revision 與跳階操作拒絕，讀取目前狀態供網路失敗後恢復；不改寫已鎖定答案。
- 不啟用 camera、cursor telemetry、gaze、模型或 GPU，不連絡或收錄外部參與者。

## 本機啟動

在此 checkout 使用 Python 3.11 的 `.venv`：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-learner-workspace.txt
$env:CUDA_VISIBLE_DEVICES = ''
$env:PYTHONDONTWRITEBYTECODE = '1'
.\.venv\Scripts\python.exe -X utf8 -B -m scripts.run_learner_workspace
```

入口：`http://127.0.0.1:8097/learner-workspace`。這個輕量入口不需要整套眼動／模型依賴。

## 驗收

- [x] 先閱讀後取得問題；完成前未揭露答案或解釋。
- [x] 鎖定答案、revision 衝突、跨 session 隔離與 TTL／刪除。
- [x] IDK、未提供信心、複習偏好與 correctness 分開。
- [x] 完整流程可產生十題、六個目標詞、三個分項的描述性檔案。
- [x] 所選複習清單可回看完整語境與解釋；沒有不實能力／保留數值。
- [x] 原 objective walkthrough 相容測試、瀏覽器操作與 HTTP 啟動確認。

實際驗證結果：

- Python 3.11.15、Flask 3.1.3；新增 learner workspace 17 項與原 objective walkthrough 16 項，合計 **33/33 PASS**。
- `node --check web/static/learner_workspace.js` 通過。
- `GET /api/learner-workspace/health` 與 `/learner-workspace` 皆為 HTTP 200，實際回應包含 no-store 與 CSP。
- 瀏覽器完整走完四個閱讀區段與十題：刻意使用答對、答錯、IDK、信心 75%、略過信心、願意／不願意／略過複習等測試輸入，確認分項與六詞紀錄一致。
- 漏答時提示並聚焦第一個選項；IDK 直接進入複習偏好；完成後自選複習篩選只留下所選一題，展開能看到原文、答案、解釋、自評與時間。
- 瀏覽器重新開始回到首頁並清空本次測試紀錄；瀏覽器 console 沒有 error/warn。
- 以上均為工程測試資料與桌面 UI 檢查，沒有受試者資料、測量效度或學習成效證據；未進行手機實機驗證。

測試指令：

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
$env:PYTHONDONTWRITEBYTECODE = '1'
.\.venv\Scripts\python.exe -X utf8 -B -m unittest scripts.test_learner_workspace scripts.test_objective_walkthrough
node --check web/static/learner_workspace.js
```

原題庫 SHA-256：`D80581E1C5A98E04FC7BE424F7B63C198B2A0292252B698625EBBC42150E6D15`；本輪沒有修改原題庫、North Star 或既有研究紀錄。

## 2026-10-26 增量：可攜式本機作答紀錄

本輪沿用既有閱讀與鎖答流程。完成後可下載 JSON，首頁可重新開啟同格式的已完成紀錄，回顧分項、原文、答案、解釋與自選複習。檔案由使用者保留；重新開啟只載入當頁記憶體，不寫資料庫、不建立或修改 session、不作 predictor 或研究資料。

- 保留客觀選項 ID、IDK、未提供自評的原因、例題來源／版本／內容雜湊與 server 時間。
- 各閱讀區段結束時間與整次第一題出現前的截止時間分開保存；模型／features 未收集，明示 `not_collected`，不宣稱已有合格 predictor snapshot。
- 僅接受有界大小、固定 schema、必要型別與內容一致的完整紀錄；錯誤檔不影響目前進度，所有外來文字以純文字渲染。
- 完成條件：完成前不可匯出、匯出／重開可回顧、壞檔不污染 session、IDK 與 missing 不混淆、無 token；重用既有輕量測試環境，不啟動服務或模型。

本輪基線已保存於 `.tmp/reading-records-20261026-baseline/`，因核心檔案原本未追蹤，使用逐檔備份與 SHA-256 核對本輪差異。

### 已實作的使用流程與資料格式

1. 完成十題後，按「下載本次作答紀錄」，由瀏覽器下載 `lexigaze-reading-YYYY-MM-DD.json`。
2. 首頁或完成頁可重新開啟檔案，標示「歷史紀錄 · 本機檔案提供，未驗證」，沿用分項、原文、作答、解釋與自選複習介面。
3. 從歷史回顧返回原本的本次紀錄或首頁。正在閱讀／作答時無法載入歷史檔案；匯入不呼叫 API、不刪除或修改目前 session、不向新作答提供 predictors。

格式固定為 `schema_version=lexigaze-reading-record-v1`，包含 instrument、measurement 與原本的描述性 profile。選項 ID、IDK、略過自評原因與 server 時間均保留。`profile.persisted=false` 指來源 server 未持久保存；下載後的檔案由使用者自行保留，重新開始不會刪除檔案。

`instrument.content_sha256` 是題庫經 UTF-8、排序鍵、無多餘空白 JSON 正規化的雜湊；各區段 `content_sha256` 是原文 UTF-8 bytes 的雜湊，匯入時核對。這些雜湊不能認證檔案來源或作答真實性；重新開啟的紀錄仍未驗證，也不能當作正式研究資料。

`measurement.first_question_cutoff_at` 僅保存第一次題目出現前的 server 時間，不因後續閱讀或作答更新。`reading_blocks` 另列各區段的閱讀結束時間；後續區段並不屬於第一次截止前的資料。`predictor_status=not_collected`、`cutoff_scope=timestamp_only_not_qualified_predictor_snapshot` 明示本輪尚無模型／features 或合格 predictor snapshot。

匯入僅支援此版四區段、十題完整例題紀錄，上限 256 KiB。schema、型別、時間順序、IDK／missing、分項計數或原文雜湊不符即拒絕；舊版缺欄位及未知版本亦拒絕。先完整檢查才換成歷史畫面，錯誤檔保留目前畫面與進度。外來文字均以 `textContent` 呈現。

### 本輪驗證與限制

本輪沿用既有 `.venv` 與 Node，未安裝環境、啟動服務、模型或相機。驗證以原 33 項及三項新增的合成案例為範圍；新增案例涵蓋完成前不提供匯出、首次截止時間、真實軟體輸出的 JSON roundtrip，以及本機檔案 parser 與 UI 事件合成檢查。UI 事件檢查包含下載內容、重開／回到目前紀錄、壞檔與過大檔不換頁、進行中禁止匯入、不向 API 傳送檔案、清除舊錯誤重試與純文字渲染。

本輪不重用下方 2026-09-07 的真實瀏覽器結果作為新 UI 的驗證；沒有本輪實際瀏覽器、檔案下載／開啟或手機人工驗收。合成 UI 事件檢查也不是瀏覽器版面、真實檔案選擇器或使用者接受證據。

實際結果：`python -X utf8 -B -m unittest -v scripts.test_learner_workspace scripts.test_objective_walkthrough` **36/36 PASS**（suite 0.609 秒）；`node --check web/static/learner_workspace.js` exit 0。完整 stdout 與 exact command 保存於 `.tmp/reading-records-20261026-evidence/`。本輪題庫原始 bytes SHA-256 仍為 `D80581E1C5A98E04FC7BE424F7B63C198B2A0292252B698625EBBC42150E6D15`。

### 本輪 North-Star Check

North-Star level: L0 engineering prerequisite。

Causal link advanced: ordered objective observations → user-managed local record → later review of the same completed evidence。

Evidence produced: 合成軟體 roundtrip 與必要回歸檢查；詳細指令與本輪結果見 `.tmp/reading-records-20261026-evidence/`。

Claims still prohibited: 題庫／profile 效度、跨次模型更新、合格 predictor snapshot、英文／認知能力、gaze utility、個人化協助與 learning/retention effect。

GPU status and reason: `not_applicable`；資料格式、狀態機與本機檔案回顧。

Next smallest step: owner 以本版實際閱讀、下載並重新開啟一份紀錄核對需求；後續再決定正式題庫效度工作或跨次 profile 的獨立規格，不能以匯入歷史檔代替模型更新。

## 2026-09-07 North-Star Check（原版歷史證據）

North-Star level: L0 engineering prerequisite；研究層級沒有因本次軟體開發升級。

Causal link advanced: reading → ordered responses and reflections → interpretable learner evidence → user-selected review。

Evidence produced: 33 項軟體測試通過、實際 HTTP 200 與完整桌面瀏覽器操作；建立單次描述性 profile 的工程基礎。

Claims still prohibited: item/profile validity、一般英文或認知能力、gaze utility、個人化協助效益、learning/retention effect。

GPU status and reason: `not_applicable`；狀態機、UI、資料結構與小型軟體測試。

Next smallest step: 用這個可操作版本檢查產品流程是否符合 owner 需求；下一版優先定義跨次保存、紀錄修正／刪除與再次閱讀時如何更新同一份 profile，再接延後檢查或獨立評量。

## 2026-10-26 main 基底整合驗證

發布版本以重新 fetch 的 GitHub `main`（`b22b63728e79cdbf68d2b85385f33089aac4ac30`）為基底，在 `feat/portable-reading-records` 整合。範圍限定為可獨立啟動與驗證本功能的 19 個檔案，未整合原開發分支另外 93 個歷史 commits；main 原有的 `web/__init__.py` 保持不變。

來源分成兩組：

- 原開發 checkout 的 10 個 learner workspace 檔案：core、專用 app、HTML/CSS/JS、launcher、輕量 requirements、兩個測試檔與本文。其中六檔為此次增量，四檔為先前已有但尚未納入 Git 的必要依賴。
- `93d845d` 已追蹤的九個 objective walkthrough 基礎檔案：core、例題 JSON、專用 app、surface policy、route、HTML/CSS/JS 與原回歸測試。這些檔案不帶入 participant store、私有資料或實驗產物。

逐檔複製後 19/19 SHA-256 與來源相同；僅本文隨後追加本節整合說明。驗證在上述 main 基底的 release checkout 執行，沿用原開發 checkout 既有 Python 環境，未安裝環境或啟動服務：

```powershell
<existing-python> -X utf8 -B -m unittest -v scripts.test_learner_workspace scripts.test_objective_walkthrough
node --check web/static/learner_workspace.js
<existing-python> -X utf8 -B -m scripts.run_learner_workspace --help
```

結果為 **36/36 PASS**（suite 1.100 秒），JS syntax 與 launcher `--help` 均 exit 0，未發現缺少依賴或 main 相容性失敗。`--help` 不啟動伺服器。本輪仍未進行新 UI 的真實瀏覽器或人工驗收。

checksum manifest、exact command 與 stdout 保存於原開發 checkout 的 `.tmp/reading-records-20261026-evidence/release-*`，不納入發布 payload。本文前段的未追蹤狀態、0.609 秒及基線路徑均指原開發 checkout 的歷史狀態；本節才是此 main 基底整合版本的驗證記錄。
