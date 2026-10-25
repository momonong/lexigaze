"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const tools = require("../web/static/learner_workspace.js");
assert.equal(typeof tools.parseReadingRecord, "function", "completed records need a bounded local-file parser");

function browserHarness(initialState, failStart = false) {
  const nodes = new Map();
  class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.listeners = {}; this.attributes = {}; this.classList = { toggle() {} }; this.dataset = {}; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    addEventListener(name, callback) { this.listeners[name] = callback; }
    setAttribute(name, value) { this.attributes[name] = value; }
    removeAttribute(name) { delete this.attributes[name]; }
    focus() {} scrollIntoView() {} remove() {}
    click() { return this.listeners.click?.(); }
    set innerHTML(_value) { throw new Error("file content must use text rendering"); }
  }
  const get = (id) => { if (!nodes.has(id)) nodes.set(id, new Element(id)); return nodes.get(id); };
  const calls = [];
  const blobs = [];
  const context = {
    TextEncoder, Blob, crypto: globalThis.crypto, AbortController, Intl,
    document: { getElementById: get, createElement: (tag) => new Element(tag), querySelectorAll: () => [], body: new Element("body") },
    window: { addEventListener() {}, requestAnimationFrame: (callback) => callback(), scrollTo() {}, setTimeout: () => 1, clearTimeout() {} },
    URL: { createObjectURL: (blob) => { blobs.push(blob); return "blob:local-record"; }, revokeObjectURL() {} },
    fetch: async (path, options) => {
      calls.push({ path, options });
      if (failStart) throw new Error("local API unavailable");
      return { ok: true, json: async () => ({ ok: true, session_id: "test-capability-not-exported", state: initialState }) };
    },
  };
  vm.runInNewContext(fs.readFileSync(require.resolve("../web/static/learner_workspace.js"), "utf8"), context);
  return {
    get, calls, blobs,
    fire: (id) => get(id).listeners.click?.(),
    open: (text, size = new TextEncoder().encode(text).length) => {
      const input = get("record-file-input");
      input.files = [{ size, text: async () => text }];
      return input.listeners.change({ target: input });
    },
  };
}

(async () => {
  const source = fs.readFileSync(0, "utf8");
  const original = JSON.parse(source);
  const before = JSON.stringify(original);
  const record = await tools.parseReadingRecord(source);
  assert.deepEqual(record, original);
  assert.equal(JSON.stringify(original), before);
  assert.equal(record.profile.observations[0].is_idk, true);
  assert.equal(record.profile.observations[0].confidence, null);
  assert.equal(record.profile.observations[1].confidence_reason, "skipped");
  assert.equal(record.measurement.predictor_status, "not_collected");
  const changed = (mutate) => { const value = JSON.parse(source); mutate(value); return JSON.stringify(value); };
  const malformed = [
    "{}", "[1]", "{", source + " ".repeat(tools.MAX_RECORD_BYTES),
    changed((value) => { value.schema_version = "future-version"; }),
    changed((value) => { value.session_id = "must-not-import-a-capability"; }),
    changed((value) => { value.profile = null; }),
    changed((value) => { value.profile.observations.pop(); }),
    changed((value) => { value.profile.observations[0].correct = true; }),
    changed((value) => { value.profile.observations[1].confidence = "75"; }),
    changed((value) => { value.profile.observations[1].review_requested = "yes"; }),
    changed((value) => { value.profile.observations[1].selected_option_id = null; }),
    changed((value) => { value.measurement.predictor_status = "qualified"; }),
    changed((value) => { value.measurement.first_question_cutoff_at = value.completed_at; }),
    changed((value) => { value.measurement.reading_blocks[0].context_text += " changed"; }),
    changed((value) => { value.profile.dimensions[0].correct += 1; }),
  ];
  for (const text of malformed) await assert.rejects(() => tools.parseReadingRecord(text));
  const literalMarkup = changed((value) => { value.profile.observations[0].explanation_zh = "<img src=x onerror=alert(1)>"; });
  assert.equal((await tools.parseReadingRecord(literalMarkup)).profile.observations[0].explanation_zh, "<img src=x onerror=alert(1)>");
  assert.equal(JSON.stringify(original), before, "rejected files must not modify the existing record");

  const history = browserHarness();
  await history.open(literalMarkup);
  assert.equal(history.calls.length, 0, "opening a local file must not contact any API");
  assert.equal(history.get("profile-heading").textContent, "回顧已完成的學習紀錄");
  assert.match(history.get("record-source").textContent, /未驗證/);
  assert.equal(history.get("item-review").children.length, 10);
  assert.equal(history.get("download-record-button").hidden, true);
  await history.open("{}");
  assert.equal(history.get("profile-heading").textContent, "回顧已完成的學習紀錄");
  assert.equal(history.get("item-review").children.length, 10);
  assert.match(history.get("record-file-status").textContent, /不是完整/);
  await history.fire("start-button");
  assert.equal(history.calls.length, 0, "history cannot start an unrelated session");
  await history.fire("filter-requested");
  assert.equal(history.get("item-review").children.length, 10);
  await history.fire("close-record-button");
  assert.equal(history.get("intro-view").hidden, false);
  assert.equal(history.get("item-review").children.length, 0);

  const state = { revision: 42, phase: "completed", progress: {}, profile: original.profile, record: original };
  const current = browserHarness(state);
  await current.fire("start-button");
  await current.fire("download-record-button");
  assert.equal(current.blobs.length, 1);
  const downloaded = await current.blobs[0].text();
  assert.deepEqual(await tools.parseReadingRecord(downloaded), original);
  assert.equal(downloaded.includes("test-capability-not-exported"), false);
  await current.open(source);
  await current.open("{}", tools.MAX_RECORD_BYTES + 1);
  await current.fire("close-record-button");
  assert.equal(current.get("profile-heading").textContent, "我的本次學習紀錄");
  assert.equal(current.get("download-record-button").hidden, false);
  assert.equal(current.calls.length, 1, "history must not rewrite, delete or advance the current session");
  assert.equal(JSON.stringify(state.record), before);

  const active = browserHarness({ revision: 0, phase: "reading", progress: { answered: 0, total: 10, block_index: 1, block_total: 4 }, context: { kind: "passage", title: "Title", text: "Text" } });
  await active.fire("start-button");
  await active.open(source);
  assert.equal(active.get("task-view").hidden, false);
  assert.equal(active.calls.length, 1, "a reading session cannot be replaced by a file");

  const failed = browserHarness(null, true);
  await failed.fire("start-button");
  assert.equal(failed.get("error-panel").hidden, false);
  await failed.open(source);
  assert.equal(failed.get("error-panel").hidden, true, "accepted history clears stale API retries");
  await failed.fire("retry-button");
  assert.equal(failed.calls.length, 1);
  process.stdout.write("record roundtrip and rejection checks passed\n");
})().catch((error) => { process.stderr.write(String(error.stack) + "\n"); process.exitCode = 1; });
