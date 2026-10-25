(() => {
  "use strict";

  const MAX_RECORD_BYTES = 256 * 1024;

  async function parseReadingRecord(text) {
    const invalid = () => { throw new Error("這不是完整的 LexiGaze v1 作答紀錄，或檔案內容已變更。請選擇完成作答後下載的 JSON。"); };
    const need = (condition) => { if (!condition) invalid(); };
    const object = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
    const string = (value, max = 12000) => typeof value === "string" && value.length > 0 && value.length <= max;
    const time = (value) => { need(string(value, 64) && /^\d{4}-\d{2}-\d{2}T/.test(value) && Number.isFinite(Date.parse(value))); return Date.parse(value); };
    const hash = (value) => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
    const keys = (value, allowed) => need(object(value) && Object.keys(value).length === allowed.length && allowed.every((key) => Object.hasOwn(value, key)));
    need(typeof text === "string" && new TextEncoder().encode(text).length <= MAX_RECORD_BYTES);
    let record;
    try { record = JSON.parse(text); } catch (_error) { invalid(); }
    keys(record, ["schema_version", "kind", "created_at", "completed_at", "instrument", "measurement", "profile", "research_evidence", "import_policy"]);
    need(record.schema_version === "lexigaze-reading-record-v1" && record.kind === "completed_example_reading_record");
    need(record.research_evidence === false && record.import_policy === "local_file_review_only_not_authenticated");
    const created = time(record.created_at);
    const completed = time(record.completed_at);
    need(created <= completed);
    const instrument = record.instrument;
    keys(instrument, ["id", "schema_version", "status", "content_sha256", "hash_encoding"]);
    need(instrument.id === "lexigaze-objective-reading-walkthrough-v1" && instrument.schema_version === 1);
    need(instrument.status === "developer_preview_only_unvalidated_items" && hash(instrument.content_sha256) && instrument.hash_encoding === "utf8_sorted_compact_json");
    const measurement = record.measurement;
    keys(measurement, ["first_question_cutoff_at", "predictor_status", "cutoff_scope", "reading_blocks"]);
    need(measurement.predictor_status === "not_collected" && measurement.cutoff_scope === "timestamp_only_not_qualified_predictor_snapshot");
    need(Array.isArray(measurement.reading_blocks) && measurement.reading_blocks.length === 4);
    let previousCutoff = created;
    for (const [index, block] of measurement.reading_blocks.entries()) {
      keys(block, ["block_index", "started_at", "cutoff_at", "context_title", "context_text", "content_sha256"]);
      need(block.block_index === index + 1 && string(block.context_title, 512) && string(block.context_text) && hash(block.content_sha256));
      need(previousCutoff <= time(block.started_at) && time(block.started_at) <= time(block.cutoff_at) && time(block.cutoff_at) <= completed);
      previousCutoff = time(block.cutoff_at);
      const digest = await globalThis.crypto.subtle.digest("SHA-256", new TextEncoder().encode(block.context_text));
      need(Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("") === block.content_sha256);
    }
    need(measurement.first_question_cutoff_at === measurement.reading_blocks[0].cutoff_at);
    const profile = record.profile;
    need(object(profile) && profile.scope === "this_session_only" && profile.status === "developer_preview_observations");
    need(profile.research_evidence === false && profile.persisted === false && profile.uncertainty === "single_session_unvalidated_items");
    need(time(profile.generated_at) === completed && JSON.stringify(profile.unavailable) === JSON.stringify(["delayed_retention", "gaze", "cognitive_ability"]));
    need(object(profile.reading) && profile.reading.completed_blocks === 4 && Number.isInteger(profile.reading.elapsed_ms) && profile.reading.elapsed_ms >= 0);
    need(profile.reading.timing_interpretation === "server_observed_reading_phase_duration_not_attention_or_reading_speed");
    need(Array.isArray(profile.observations) && profile.observations.length === 10 && Array.isArray(profile.dimensions) && profile.dimensions.length === 3);
    const counts = { contextual_meaning: { planned: 6, correct: 0, answered: 0, idk_count: 0 }, passage_comprehension: { planned: 2, correct: 0, answered: 0, idk_count: 0 }, novel_context_application: { planned: 2, correct: 0, answered: 0, idk_count: 0 } };
    const itemIds = new Set();
    for (const item of profile.observations) {
      need(object(item) && string(item.item_id, 128) && !itemIds.has(item.item_id) && Object.hasOwn(counts, item.kind));
      itemIds.add(item.item_id);
      need(Number.isInteger(item.block_index) && item.block_index >= 1 && item.block_index <= 4);
      const block = measurement.reading_blocks[item.block_index - 1];
      need(item.context_title === block.context_title && item.context_text === block.context_text && item.reading_started_at === block.started_at && item.reading_ended_at === block.cutoff_at);
      need(item.target === null || string(item.target, 512));
      for (const key of ["prompt", "selected_text", "selected_option_id", "correct_text", "explanation_zh"]) need(string(item[key]));
      need(typeof item.correct === "boolean" && typeof item.is_idk === "boolean");
      need(time(item.answer_locked_at) >= time(block.cutoff_at) && time(item.answer_locked_at) <= completed && item.observed_at === item.answer_locked_at);
      if (item.is_idk) {
        need(item.correct === false && item.confidence === null && item.confidence_reason === "idk_not_applicable" && item.confidence_observed_at === null);
      } else {
        need((item.confidence === null && item.confidence_reason === "skipped") || ([25, 50, 75, 100].includes(item.confidence) && item.confidence_reason === "provided"));
        need(time(item.confidence_observed_at) >= time(item.answer_locked_at) && time(item.confidence_observed_at) <= completed);
      }
      need((item.review_requested === null && item.review_reason === "skipped") || (typeof item.review_requested === "boolean" && item.review_reason === "provided"));
      need(time(item.review_observed_at) >= time(item.confidence_observed_at || item.answer_locked_at) && time(item.review_observed_at) <= completed);
      need(item.instrument_id === instrument.id && item.instrument_schema_version === instrument.schema_version && item.instrument_status === instrument.status);
      need(item.source === "learner_workspace_v1/objective_option_response" && item.workspace_version === "lexigaze-reading-learner-workspace-v1" && item.evidence_status === "unvalidated_example_item_observation");
      counts[item.kind].answered += 1;
      counts[item.kind].correct += Number(item.correct);
      counts[item.kind].idk_count += Number(item.is_idk);
    }
    const dimensionKeys = new Set();
    for (const dimension of profile.dimensions) {
      need(object(dimension) && Object.hasOwn(counts, dimension.key) && !dimensionKeys.has(dimension.key) && string(dimension.label, 128));
      dimensionKeys.add(dimension.key);
      for (const key of ["planned", "correct", "answered", "idk_count"]) need(dimension[key] === counts[dimension.key][key]);
      need(dimension.answered === dimension.planned);
    }
    return record;
  }

  // The same bounded parser is reusable from a local Node consumer without a DOM.
  if (typeof module === "object" && module.exports) {
    module.exports = { MAX_RECORD_BYTES, parseReadingRecord };
    return;
  }

  const api = "/api/learner-workspace";
  const byId = (id) => document.getElementById(id);
  const session = { id: null, state: null, busy: false, retry: null, filter: "all", history: null, historyBusy: false };
  const views = { intro: byId("intro-view"), task: byId("task-view"), profile: byId("profile-view") };
  const confidenceLabels = { 25: "25% · 比較像猜測", 50: "50% · 有點不確定", 75: "75% · 大致有把握", 100: "100% · 很有把握" };
  const kindLabels = { contextual_meaning: "語境字義", passage_comprehension: "文章理解", novel_context_application: "新語境應用" };

  function node(tag, className, text) {
    const result = document.createElement(tag);
    if (className) result.className = className;
    if (text !== undefined && text !== null) result.textContent = String(text);
    return result;
  }

  function english(tag, className, text) {
    const result = node(tag, className, text);
    result.lang = "en";
    return result;
  }

  function focus(element) {
    element.tabIndex = -1;
    window.requestAnimationFrame(() => element.focus({ preventScroll: true }));
  }

  function showView(name) {
    Object.entries(views).forEach(([key, view]) => { view.hidden = key !== name; });
    window.scrollTo({ top: 0, behavior: "auto" });
  }

  function setBusy(busy, message = "") {
    session.busy = busy;
    byId("workspace").setAttribute("aria-busy", String(busy));
    document.querySelectorAll("main button, main input").forEach((control) => { control.disabled = busy; });
    byId("request-status").textContent = busy ? message : "";
  }

  function clearError() {
    byId("error-panel").hidden = true;
    byId("error-message").textContent = "";
    session.retry = null;
  }

  function showError(error, retry) {
    session.retry = retry;
    byId("error-panel").hidden = false;
    byId("error-message").textContent = error.message || "本機連線暫時中斷，請重新確認目前進度。";
    byId("retry-button").hidden = !retry;
    focus(byId("error-heading"));
    byId("error-panel").scrollIntoView({ block: "nearest", behavior: "auto" });
  }

  async function request(path, { method = "GET", body, useSession = true } = {}) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 15000);
    const headers = { Accept: "application/json" };
    if (useSession && session.id) headers["X-Learner-Session"] = session.id;
    if (body !== undefined) headers["Content-Type"] = "application/json";
    try {
      const response = await fetch(`${api}${path}`, {
        method, headers, body: body === undefined ? undefined : JSON.stringify(body),
        credentials: "omit", cache: "no-store", signal: controller.signal,
      });
      let payload;
      try { payload = await response.json(); } catch (_error) {
        throw new Error("沒有收到完整回應。請重新確認進度後繼續。");
      }
      if (!response.ok || payload.ok === false) {
        const error = new Error(response.status === 404 || response.status === 410
          ? "這次暫存紀錄已到期或被清空。請回到開始，建立新的閱讀紀錄。"
          : response.status === 409
            ? "進度已更新，正在重新確認目前的步驟。"
            : "這一步暫時無法完成，請重新確認進度後再試一次。");
        error.status = response.status;
        throw error;
      }
      return payload;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("本機回應時間較長。請重新確認進度；已送出的答案不會被重複記錄。");
      throw error;
    } finally { window.clearTimeout(timeout); }
  }

  function acceptState(state) {
    if (!state || !["reading", "answering", "confidence", "review", "completed"].includes(state.phase)
      || !Number.isInteger(state.revision) || !state.progress) {
      throw new Error("目前進度的格式不完整，請重新確認，或清空後重新開始。");
    }
    session.state = state;
    renderState();
  }

  async function start() {
    if (session.busy || session.historyBusy || session.history) return;
    clearError();
    setBusy(true, "正在準備本次閱讀…");
    try {
      const payload = await request("/session", { method: "POST", body: {}, useSession: false });
      if (!payload.session_id) throw new Error("無法建立本次紀錄，請重新開始。");
      session.id = payload.session_id;
      session.filter = "all";
      acceptState(payload.state);
    } catch (error) { showError(error, session.id ? () => recover(null) : start); }
    finally { setBusy(false); }
  }

  async function recover(pending) {
    if (session.busy) return;
    clearError();
    setBusy(true, "正在確認已保存的作答進度…");
    let replay = false;
    try {
      const payload = await request("/state");
      acceptState(payload.state);
      replay = Boolean(pending && payload.state.revision === pending.revision);
    } catch (error) {
      showError(error, [404, 410].includes(error.status) ? null : () => recover(pending));
    } finally { setBusy(false); }
    if (replay) await advance(pending);
  }

  async function advance(action) {
    if (session.busy || !session.state) return;
    const pending = { revision: session.state.revision, ...action };
    clearError();
    setBusy(true, "正在確認這一步…");
    try {
      const payload = await request("/advance", { method: "POST", body: pending });
      acceptState(payload.state);
    } catch (error) {
      if (error.status === 409) {
        try { acceptState((await request("/state")).state); }
        catch (syncError) { showError(syncError, () => recover(pending)); }
      } else {
        showError(error, [404, 410].includes(error.status) ? null : () => recover(pending));
      }
    } finally { setBusy(false); }
  }

  async function restart() {
    if (session.busy || session.historyBusy) return;
    clearError();
    setBusy(true, "正在清空本次紀錄…");
    try {
      if (session.id) {
        try { await request("/session", { method: "DELETE" }); }
        catch (error) { if (![404, 410].includes(error.status)) throw error; }
      }
      session.id = null;
      session.state = null;
      session.filter = "all";
      session.history = null;
      byId("record-file-status").textContent = "";
      ["task-content", "dimension-cards", "profile-summary", "word-evidence", "item-review"].forEach((id) => byId(id).replaceChildren());
      showView("intro");
      focus(byId("intro-heading"));
    } catch (error) { showError(error, restart); }
    finally { setBusy(false); }
  }

  function visibleProfile() {
    return session.history ? session.history.profile : session.state?.profile;
  }

  function chooseRecord() {
    if (session.busy || session.historyBusy || (session.state && session.state.phase !== "completed")) return;
    byId("record-file-input").value = "";
    byId("record-file-input").click();
  }

  async function openRecord(event) {
    const file = event.target.files?.[0];
    if (!file || session.busy || session.historyBusy || (session.state && session.state.phase !== "completed")) return;
    session.historyBusy = true;
    setBusy(true, "正在檢查本機紀錄…");
    byId("record-file-status").textContent = "";
    try {
      if (file.size > MAX_RECORD_BYTES) throw new Error("紀錄檔超過 256 KiB，請選擇原本下載的 JSON。");
      const candidate = await parseReadingRecord(await file.text());
      clearError();
      session.history = candidate;
      session.filter = "all";
      renderProfile(candidate.profile);
      byId("record-file-status").textContent = "已開啟本機檔案，只供歷史回顧；來源與作答未經驗證。";
    } catch (error) {
      byId("record-file-status").textContent = error.message || "無法開啟這份紀錄，目前的進度與畫面保持原樣。";
    } finally {
      session.historyBusy = false;
      event.target.value = "";
      setBusy(false);
    }
  }

  function closeRecord() {
    if (session.busy || session.historyBusy) return;
    session.history = null;
    session.filter = "all";
    byId("record-file-status").textContent = "";
    if (session.state?.phase === "completed") renderState();
    else {
      ["dimension-cards", "profile-summary", "word-evidence", "item-review"].forEach((id) => byId(id).replaceChildren());
      showView("intro"); focus(byId("intro-heading"));
    }
  }

  function downloadRecord() {
    if (session.busy || session.history || session.state?.phase !== "completed" || !session.state.record) return;
    const text = JSON.stringify(session.state.record, null, 2);
    const blob = new Blob([text + "\n"], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = node("a");
    link.href = url;
    link.download = `lexigaze-reading-${session.state.record.completed_at.slice(0, 10)}.json`;
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    byId("record-file-status").textContent = "已交給瀏覽器下載。下次可從首頁重新開啟這份 JSON，回顧作答與複習清單。";
  }

  function button(text, handler, className = "button button-primary") {
    const result = node("button", className, text);
    result.type = "button";
    result.addEventListener("click", handler);
    return result;
  }

  function contextCard(context, { expanded = false } = {}) {
    const details = node("details", "context-card");
    details.open = expanded;
    details.append(node("summary", "", "回看閱讀內容"));
    const body = node("div", "context-body");
    body.append(english("h3", "", context.title), english("p", "reading-text", context.text));
    details.append(body);
    return details;
  }

  function heading(title, subtitle) {
    const wrapper = node("div", "task-heading");
    const h1 = node("h1", "", title);
    h1.id = "task-heading";
    wrapper.append(h1);
    if (subtitle) wrapper.append(node("p", "", subtitle));
    return wrapper;
  }

  function radioGroup(name, options, legendText) {
    const fieldset = node("fieldset", "choice-group");
    fieldset.append(node("legend", "choice-legend", legendText));
    options.forEach((option) => {
      const label = node("label", "choice");
      const input = node("input");
      input.type = "radio";
      input.name = name;
      input.value = option.value;
      const text = node("span", "", option.text);
      if (option.lang) text.lang = option.lang;
      label.append(input, text);
      fieldset.append(label);
    });
    return fieldset;
  }

  function choiceActions(group, text, makeAction, skipAction) {
    const actions = node("div", "task-actions");
    const message = node("p", "choice-error");
    message.setAttribute("role", "alert");
    actions.append(button(text, () => {
      const selected = group.querySelector("input:checked");
      if (!selected) {
        message.textContent = "請先選擇一個選項。";
        group.querySelector("input")?.focus();
        return;
      }
      advance(makeAction(selected.value));
    }));
    if (skipAction) actions.append(button("略過這一步", () => advance(skipAction), "button button-quiet"));
    group.addEventListener("change", () => { message.textContent = ""; });
    const wrapper = node("div");
    wrapper.append(message, actions);
    return wrapper;
  }

  function renderState() {
    const state = session.state;
    if (state.phase === "completed") { renderProfile(state.profile); return; }
    showView("task");
    const progress = state.progress;
    byId("block-label").textContent = `閱讀 ${progress.block_index} / ${progress.block_total}`;
    byId("task-progress").textContent = `已作答 ${progress.answered} / ${progress.total} 題`;
    const completed = Math.max(0, Math.min(10, Number(progress.answered) || 0));
    byId("progress-fill").className = `progress-fill progress-${completed}`;
    byId("progress-indicator").setAttribute("aria-valuenow", String(completed));
    document.querySelectorAll("[data-phase]").forEach((item) => {
      const active = item.dataset.phase === state.phase;
      item.classList.toggle("current", active);
      if (active) item.setAttribute("aria-current", "step"); else item.removeAttribute("aria-current");
    });
    const content = byId("task-content");
    content.replaceChildren();

    if (state.phase === "reading") {
      content.append(heading(state.context.kind === "novel_context" ? "換個情境，再讀一次" : "先給自己一段閱讀時間", "先閱讀內容。準備好後，再進入作答。"));
      const article = node("article", "reading-card");
      article.append(english("h2", "reading-title", state.context.title), english("p", "reading-text", state.context.text));
      content.append(article);
      const actions = node("div", "task-actions");
      actions.append(button("我讀完了，開始作答", () => advance({ action: "complete_reading" })), node("p", "quiet-note", "作答時仍可以回看內容。"));
      content.append(actions);
    } else if (state.phase === "answering") {
      content.append(heading("你會怎麼回答？", "選擇最符合語境的答案；不確定時，也可以選擇 I do not know。"));
      if (state.context) content.append(contextCard(state.context));
      const question = node("div", "question-panel");
      question.append(english("h2", "question-prompt", state.question.prompt));
      const choices = radioGroup("answer", state.question.options.map((option) => ({ value: option.option_id, text: option.text, lang: "en" })), "選擇你的答案");
      question.append(choices, choiceActions(choices, "確認並鎖定答案", (value) => ({ action: "answer", option_id: value })));
      question.append(node("p", "quiet-note", "確認後無法回頭修改；所有題目完成後，一起核對答案。"));
      content.append(question);
    } else if (state.phase === "confidence") {
      content.append(heading("對剛才的答案，有多少把握？", "這是你當下的感受，和答案是否正確分開記錄。"));
      renderLockedAnswer(content, state);
      const choices = radioGroup("confidence", [25, 50, 75, 100].map((value) => ({ value: String(value), text: confidenceLabels[value] })), "選擇把握程度");
      content.append(choices, choiceActions(choices, "記下把握程度", (value) => ({ action: "confidence", value: Number(value) }), { action: "confidence", value: null }));
    } else if (state.phase === "review") {
      content.append(heading("這一題，想留到最後再看嗎？", "依你的需要選擇。這份複習清單由你決定。"));
      renderLockedAnswer(content, state);
      const choices = radioGroup("review", [{ value: "yes", text: "想，加入我的複習清單" }, { value: "no", text: "這一題先不用" }], "選擇是否複習");
      content.append(choices, choiceActions(choices, "確認並繼續", (value) => ({ action: "review", value: value === "yes" }), { action: "review", value: null }));
    }
    if (session.busy) setBusy(true, byId("request-status").textContent);
    focus(byId("task-heading"));
  }

  function renderLockedAnswer(content, state) {
    const card = node("div", "locked-answer");
    card.append(node("p", "eyebrow", "已鎖定的作答"));
    if (state.question?.prompt) card.append(english("p", "locked-prompt", state.question.prompt));
    card.append(english("strong", "", state.selected_text || "已記錄答案"));
    content.append(card);
    if (state.context) content.append(contextCard(state.context));
  }

  function formatTime(value) {
    const date = new Date(value);
    if (Number.isNaN(date.valueOf())) return "本次作答";
    return new Intl.DateTimeFormat("zh-TW", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(date);
  }

  function confidenceText(item) {
    if (item.is_idk) return "把握程度：選擇不知道，未詢問";
    return item.confidence === null || item.confidence === undefined ? "把握程度：略過" : `把握程度：${item.confidence}%`;
  }

  function reviewText(item) {
    return item.review_requested === true ? "想複習" : item.review_requested === false ? "暫不複習" : "複習意願：略過";
  }

  function outcomePill(item) {
    return node("span", `outcome-pill ${item.correct ? "correct" : item.is_idk ? "unknown" : "incorrect"}`, item.correct ? "答對" : item.is_idk ? "不知道" : "未答對");
  }

  function renderProfile(profile) {
    if (!profile || !Array.isArray(profile.dimensions) || !Array.isArray(profile.observations)) throw new Error("作答已完成，但紀錄尚未完整載入。請重新確認。");
    showView("profile");
    const historical = Boolean(session.history);
    byId("profile-heading").textContent = historical ? "回顧已完成的學習紀錄" : "我的本次學習紀錄";
    byId("record-source").textContent = historical ? "歷史紀錄 · 本機檔案提供，未驗證" : "本次紀錄 · 作答已完成";
    byId("download-record-button").hidden = historical || !session.state?.record;
    byId("restart-button").hidden = historical;
    byId("close-record-button").hidden = !historical;
    byId("close-record-button").textContent = session.state?.phase === "completed" ? "回到我的本次紀錄" : "回到首頁";
    byId("record-storage-note").textContent = historical
      ? "只在此頁回顧檔案，不會更新目前作答或個人模型；關閉回顧會返回首頁或目前紀錄。"
      : "下載後由你保留；下次可重新開啟回顧。系統不會自動保存跨次個人檔案。";
    const cards = byId("dimension-cards");
    cards.replaceChildren();
    profile.dimensions.forEach((dimension) => {
      const card = node("article", "dimension-card");
      card.append(node("p", "eyebrow", kindLabels[dimension.key] || dimension.label));
      const value = node("div", "dimension-value");
      value.append(node("strong", "", dimension.correct), node("span", "", `/ ${dimension.planned} 題答對`));
      card.append(value, node("p", "quiet-note", `已作答 ${dimension.answered} 題 · 不知道 ${dimension.idk_count} 題`));
      cards.append(card);
    });
    const requested = profile.observations.filter((item) => item.review_requested === true);
    const summary = byId("profile-summary");
    summary.replaceChildren();
    const summaryCopy = node("div");
    summaryCopy.append(node("strong", "", `你為自己留下了 ${requested.length} 題想複習的內容`), node("p", "", `${formatTime(profile.generated_at)} · 這份紀錄只來自本次 ${profile.observations.length} 題作答。`));
    summary.append(summaryCopy, button("查看我的複習清單", () => {
      session.filter = "requested";
      renderReview(profile);
      byId("review-heading").scrollIntoView({ block: "start", behavior: "auto" });
      focus(byId("review-heading"));
    }, "button button-secondary"));
    renderWords(profile.observations);
    renderReview(profile);
    if (session.busy) setBusy(true, byId("request-status").textContent);
    focus(byId("profile-heading"));
  }

  function renderWords(observations) {
    const targets = new Map();
    observations.filter((item) => item.kind !== "passage_comprehension" && item.target).forEach((item) => {
      if (!targets.has(item.target)) targets.set(item.target, []);
      targets.get(item.target).push(item);
    });
    const grid = byId("word-evidence");
    grid.replaceChildren();
    targets.forEach((items, target) => {
      const card = node("article", "word-card");
      card.append(english("h3", "", target));
      items.forEach((item) => {
        const row = node("div", "word-observation");
        row.append(node("span", "", kindLabels[item.kind] || item.kind), outcomePill(item));
        card.append(row, node("p", "word-meta", `${confidenceText(item)} · ${reviewText(item)}`));
      });
      card.append(node("p", "evidence-note", items.length > 1 ? "本次兩個語境的作答；尚未檢查延後保留。" : "只有本次這一個語境的作答。"));
      grid.append(card);
    });
  }

  function renderReview(profile) {
    const filtered = session.filter === "requested" ? profile.observations.filter((item) => item.review_requested === true) : profile.observations;
    byId("filter-all").setAttribute("aria-pressed", String(session.filter === "all"));
    byId("filter-requested").setAttribute("aria-pressed", String(session.filter === "requested"));
    byId("review-count").textContent = session.filter === "requested" ? `顯示你自行選擇的 ${filtered.length} 題。` : `顯示全部 ${filtered.length} 題，包含你選擇不知道的題目。`;
    const list = byId("item-review");
    list.replaceChildren();
    if (!filtered.length) {
      const empty = node("div", "empty-review");
      empty.append(node("h3", "", "這次沒有選擇要複習的題目"), node("p", "", "你仍然可以從「全部題目」回看文章與解釋。"));
      list.append(empty);
      return;
    }
    filtered.forEach((item) => {
      const details = node("details", "review-item");
      const summary = node("summary");
      const title = node("span", "review-title");
      title.append(node("small", "", `${kindLabels[item.kind] || item.kind}${item.review_requested === true ? " · 我想複習" : ""}`), english("span", "", item.prompt));
      summary.append(outcomePill(item), title, node("span", "review-expand", "+"));
      const body = node("div", "review-body");
      const context = node("div", "review-context");
      context.append(english("h3", "", item.context_title), english("p", "reading-text", item.context_text));
      const selected = node("p", "answer-line");
      selected.append(node("strong", "", "你的答案"), english("span", "", item.selected_text));
      const correct = node("p", "answer-line");
      correct.append(node("strong", "", "正確答案"), english("span", "", item.correct_text));
      const explanation = node("div", "answer-explanation");
      explanation.append(node("strong", "", "回到語境看原因"), node("p", "", item.explanation_zh));
      const metadata = node("p", "observation-meta", `${confidenceText(item)} · ${reviewText(item)} · ${formatTime(item.observed_at)}`);
      body.append(context, selected, correct, explanation, metadata);
      details.append(summary, body);
      list.append(details);
    });
  }

  byId("start-button").addEventListener("click", start);
  byId("restart-button").addEventListener("click", restart);
  byId("error-reset-button").addEventListener("click", restart);
  byId("open-record-button").addEventListener("click", chooseRecord);
  byId("open-record-from-profile-button").addEventListener("click", chooseRecord);
  byId("record-file-input").addEventListener("change", openRecord);
  byId("download-record-button").addEventListener("click", downloadRecord);
  byId("close-record-button").addEventListener("click", closeRecord);
  byId("retry-button").addEventListener("click", () => { if (!session.busy && session.retry) session.retry(); });
  byId("filter-all").addEventListener("click", () => { if (!session.busy && visibleProfile()) { session.filter = "all"; renderReview(visibleProfile()); } });
  byId("filter-requested").addEventListener("click", () => { if (!session.busy && visibleProfile()) { session.filter = "requested"; renderReview(visibleProfile()); } });
  window.addEventListener("pagehide", () => {
    if (!session.id) return;
    fetch(`${api}/session`, { method: "DELETE", headers: { "X-Learner-Session": session.id, Accept: "application/json" }, credentials: "omit", cache: "no-store", keepalive: true }).catch(() => {});
  });
})();
