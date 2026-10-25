(() => {
  "use strict";

  const views = {
    intro: document.querySelector("#intro-view"),
    task: document.querySelector("#task-view"),
    result: document.querySelector("#result-view"),
    fatal: document.querySelector("#fatal-view"),
  };
  const startButton = document.querySelector("#start-button");
  const backButton = document.querySelector("#back-button");
  const nextButton = document.querySelector("#next-button");
  const restartButton = document.querySelector("#restart-button");
  const fatalReloadButton = document.querySelector("#fatal-reload-button");
  const stageContent = document.querySelector("#stage-content");
  const stageMessage = document.querySelector("#stage-message");
  const elapsedTime = document.querySelector("#elapsed-time");

  const state = {
    form: null,
    stages: [],
    stageIndex: 0,
    responses: new Map(),
    startedAt: null,
    elapsedTimer: null,
    submitting: false,
  };

  function showView(name) {
    Object.entries(views).forEach(([key, view]) => {
      view.hidden = key !== name;
    });
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text !== undefined) {
      node.textContent = text;
    }
    return node;
  }

  function formatDuration(milliseconds) {
    const seconds = Math.max(0, Math.floor(milliseconds / 1000));
    const minutes = Math.floor(seconds / 60);
    const remainder = seconds % 60;
    return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
  }

  function currentElapsedMs() {
    return state.startedAt === null ? 0 : Math.max(0, Date.now() - state.startedAt);
  }

  function startTimer() {
    window.clearInterval(state.elapsedTimer);
    elapsedTime.textContent = "00:00";
    state.elapsedTimer = window.setInterval(() => {
      elapsedTime.textContent = formatDuration(currentElapsedMs());
    }, 1000);
  }

  async function readJson(response) {
    let payload;
    try {
      payload = await response.json();
    } catch (_error) {
      throw new Error("伺服器回應不是可讀的 JSON。請停止本次流程。");
    }
    if (!response.ok) {
      throw new Error(payload.error || "本機流程拒絕了這個請求。");
    }
    return payload;
  }

  function buildStages(form) {
    const nonTransfer = form.items.filter(
      (item) => item.kind !== "novel_context_application",
    );
    const passageStages = form.passages.map((passage) => ({
      type: "passage",
      passage,
      items: passage.item_ids.map((itemId) => {
        const item = nonTransfer.find((candidate) => candidate.item_id === itemId);
        if (!item) {
          throw new Error(`題目 ${itemId} 沒有出現在公開表單中。`);
        }
        return item;
      }),
    }));
    const transferItems = form.items.filter(
      (item) => item.kind === "novel_context_application",
    );
    if (passageStages.length !== 2 || transferItems.length !== 2) {
      throw new Error("公開表單不符合兩篇短文與兩題換情境題的約定。");
    }
    return [
      ...passageStages,
      { type: "transfer", items: transferItems },
    ];
  }

  function renderQuestion(item, ordinal) {
    const card = element("fieldset", "question-card");
    card.dataset.itemId = item.item_id;
    const legend = document.createElement("legend");
    legend.append(element("span", "question-number", `Question ${ordinal} of 10`));
    if (item.context) {
      const context = element("span", "fresh-context", item.context);
      context.lang = "en";
      legend.append(context);
    }
    const prompt = element("span", "question-prompt", item.prompt);
    prompt.lang = "en";
    legend.append(prompt);
    card.append(legend);
    const options = element("div", "options");
    item.options.forEach((option) => {
      const label = element("label", "option");
      const input = document.createElement("input");
      input.type = "radio";
      input.name = item.item_id;
      input.value = option.option_id;
      input.checked = state.responses.get(item.item_id) === option.option_id;
      input.addEventListener("change", () => {
        state.responses.set(item.item_id, option.option_id);
        stageMessage.textContent = "";
      });
      const optionText = element("span", "", option.text);
      optionText.lang = "en";
      label.append(input, optionText);
      options.append(label);
    });
    card.append(options);
    return card;
  }

  function globalOrdinal(itemId) {
    return state.form.items.findIndex((item) => item.item_id === itemId) + 1;
  }

  function renderStage() {
    const stage = state.stages[state.stageIndex];
    stageContent.replaceChildren();
    stageMessage.textContent = "";
    document.querySelector("#stage-label").textContent =
      `第 ${state.stageIndex + 1} 部分，共 ${state.stages.length} 部分`;
    document.querySelector("#progress-fill").className =
      `progress-fill progress-fill--${state.stageIndex + 1}`;
    backButton.disabled = state.stageIndex === 0 || state.submitting;
    nextButton.disabled = state.submitting;
    nextButton.textContent = state.stageIndex === state.stages.length - 1
      ? "完成並看摘要"
      : "下一部分";

    if (stage.type === "passage") {
      document.querySelector("#progress-title").textContent =
        state.stageIndex === 0 ? "短文一" : "短文二";
      stageContent.append(
        element("p", "passage-label", `Passage ${state.stageIndex + 1}`),
        element("h1", "passage-title", stage.passage.title),
        element("p", "passage-text", stage.passage.text),
        element(
          "p",
          "task-note",
          "作答時可以回看短文。四題全部完成後再進到下一部分，中途不顯示答案。",
        ),
      );
      stageContent.querySelector(".passage-title").lang = "en";
      stageContent.querySelector(".passage-text").lang = "en";
    } else {
      document.querySelector("#progress-title").textContent = "換一個情境再判斷";
      stageContent.append(
        element("p", "passage-label", "Fresh contexts"),
        element("h1", "passage-title", "同一個詞，放到另一個情境"),
        element(
          "p",
          "transfer-intro",
          "請只根據每題提供的新情境作答。這兩題不代表延遲保留或學習效果。",
        ),
      );
    }

    const list = element("div", "question-list");
    stage.items.forEach((item) => {
      list.append(renderQuestion(item, globalOrdinal(item.item_id)));
    });
    stageContent.append(list);
    const stageHeading = stageContent.querySelector("h1");
    stageHeading.tabIndex = -1;
    window.requestAnimationFrame(() => stageHeading.focus({ preventScroll: true }));
  }

  function stageIsComplete() {
    return state.stages[state.stageIndex].items.every((item) =>
      state.responses.has(item.item_id),
    );
  }

  function dimensionCard(title, dimension, note) {
    const card = element("article", "score-card");
    card.append(element("p", "card-kicker", title));
    const value = element("div", "score-value");
    value.append(
      element("strong", "", String(dimension.correct)),
      element("span", "", `/ ${dimension.planned} 題`),
    );
    card.append(value, element("p", "", note));
    return card;
  }

  function renderResults(result) {
    document.querySelector("#result-time").textContent = formatDuration(
      result.total_elapsed_ms,
    );
    const cards = document.querySelector("#score-cards");
    cards.replaceChildren(
      dimensionCard(
        "語境字義",
        result.dimensions.contextual_meaning,
        "六個詞在原短文中的作答",
      ),
      dimensionCard(
        "文章理解",
        result.dimensions.passage_comprehension,
        "兩篇短文各一題，不合成整體能力",
      ),
      dimensionCard(
        "換情境再判斷",
        result.dimensions.novel_context_application,
        "同一個 session 的兩個新情境",
      ),
    );

    const paired = document.querySelector("#paired-evidence");
    paired.replaceChildren();
    result.paired_context_evidence.forEach((entry) => {
      const card = element("article", "evidence-card");
      card.append(
        element("strong", "", entry.target),
        element("p", "", entry.interpretation_zh),
      );
      paired.append(card);
    });

    const roleLabels = {
      competing_meaning: "選到另一個常見字義",
      local_detail_misread: "選到鄰近但未回答題目的細節",
      unsupported_inference: "選到短文沒有支持的推論",
      opposite_or_reversed: "選到與短文相反或顛倒的敘述",
      surface_association: "選到表面聯想選項",
    };
    const patternSummary = element("div", "pattern-summary");
    patternSummary.append(
      element("h3", "", "其他選項的類型"),
      element("p", "", "只描述你選到哪一類選項，不把它解釋成已證明的心理誤解。"),
    );
    const patternList = element("div", "pattern-list");
    const patterns = Object.entries(result.error_patterns);
    if (patterns.length === 0) {
      patternList.append(element("span", "", "沒有可列出的其他選項類型"));
    } else {
      patterns.forEach(([role, count]) => {
        patternList.append(
          element("span", "", `${roleLabels[role] || role}：${count} 題`),
        );
      });
    }
    patternSummary.append(patternList);
    paired.closest(".result-section").append(patternSummary);

    document.querySelector("#idk-summary").textContent =
      `I do not know：${result.idk_count} 題；未作答：${result.missing_count} 題`;
    const review = document.querySelector("#item-review");
    review.replaceChildren();
    result.item_results.forEach((item, index) => {
      const details = element("details", "review-item");
      const summary = document.createElement("summary");
      const pill = element(
        "span",
        `result-pill ${item.correct ? "result-pill--correct" : "result-pill--incorrect"}`,
        item.correct ? "答對" : "未答對",
      );
      const reviewPrompt = element("span", "", `${index + 1}. ${item.prompt}`);
      reviewPrompt.lang = "en";
      summary.append(
        pill,
        reviewPrompt,
        element("span", "review-chevron", "⌄"),
      );
      const body = element("div", "review-body");
      const selected = element("p");
      const selectedText = element("span", "", item.selected_text);
      selectedText.lang = "en";
      selected.append(element("strong", "", "你的答案："), selectedText);
      const correct = element("p");
      const correctText = element("span", "", item.correct_text);
      correctText.lang = "en";
      correct.append(element("strong", "", "正確答案："), correctText);
      body.append(selected, correct, element("p", "", item.explanation_zh));
      details.append(summary, body);
      review.append(details);
    });
    showView("result");
    const resultHeading = document.querySelector("#result-view h1");
    resultHeading.tabIndex = -1;
    resultHeading.focus({ preventScroll: true });
  }

  function fatal(message) {
    window.clearInterval(state.elapsedTimer);
    document.querySelector("#fatal-message").textContent = message;
    showView("fatal");
    const fatalHeading = document.querySelector("#fatal-view h1");
    fatalHeading.tabIndex = -1;
    fatalHeading.focus({ preventScroll: true });
  }

  async function startWalkthrough() {
    startButton.disabled = true;
    document.querySelector("#intro-error").textContent = "正在載入本機題目…";
    try {
      const response = await fetch("/api/objective-profile/form", {
        method: "GET",
        headers: { Accept: "application/json" },
        cache: "no-store",
        credentials: "omit",
      });
      state.form = await readJson(response);
      state.stages = buildStages(state.form);
      state.stageIndex = 0;
      state.responses.clear();
      state.startedAt = Date.now();
      startTimer();
      renderStage();
      showView("task");
    } catch (error) {
      startButton.disabled = false;
      document.querySelector("#intro-error").textContent = error.message;
    }
  }

  async function submitWalkthrough() {
    state.submitting = true;
    nextButton.disabled = true;
    nextButton.textContent = "正在核對…";
    const totalElapsedMs = currentElapsedMs();
    try {
      const response = await fetch("/api/objective-profile/score", {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        cache: "no-store",
        credentials: "omit",
        body: JSON.stringify({
          responses: Object.fromEntries(state.responses.entries()),
          total_elapsed_ms: totalElapsedMs,
        }),
      });
      const result = await readJson(response);
      window.clearInterval(state.elapsedTimer);
      renderResults(result);
    } catch (error) {
      state.submitting = false;
      nextButton.disabled = false;
      nextButton.textContent = "完成並看摘要";
      stageMessage.textContent = `無法完成計分：${error.message}`;
    }
  }

  startButton.addEventListener("click", startWalkthrough);
  backButton.addEventListener("click", () => {
    if (state.stageIndex > 0 && !state.submitting) {
      state.stageIndex -= 1;
      renderStage();
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  });
  nextButton.addEventListener("click", () => {
    if (state.submitting) {
      return;
    }
    if (!stageIsComplete()) {
      stageMessage.textContent = "請先完成這一部分的每一題；不確定時可以選 I do not know。";
      return;
    }
    if (state.stageIndex < state.stages.length - 1) {
      state.stageIndex += 1;
      renderStage();
      window.scrollTo({ top: 0, behavior: "smooth" });
      return;
    }
    submitWalkthrough();
  });
  restartButton.addEventListener("click", () => window.location.reload());
  fatalReloadButton.addEventListener("click", () => window.location.reload());

  window.addEventListener("error", (event) => {
    fatal(`頁面發生錯誤：${event.message || "unknown client error"}`);
  });
  window.addEventListener("unhandledrejection", (event) => {
    const reason = event.reason instanceof Error ? event.reason.message : String(event.reason);
    fatal(`頁面發生錯誤：${reason}`);
  });
})();
