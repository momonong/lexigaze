"""Behavioral and request-isolation tests for the ephemeral owner workspace."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from core.learner_workspace import LearnerWorkspaceError, LearnerWorkspaceStore, SESSION_TTL_SECONDS
from core.objective_profile_walkthrough import load_walkthrough_bank
from web.learner_workspace_app import create_learner_workspace_app

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8097"
API = "/api/learner-workspace"


class Clock:
    def __init__(self):
        self.value = 1_800_000_000.0

    def __call__(self):
        return self.value

    def tick(self, seconds=1):
        self.value += seconds


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.bank = load_walkthrough_bank()
        self.items = {item["item_id"]: item for item in self.bank["items"]}
        self.store = LearnerWorkspaceStore(bank=self.bank, clock=self.clock)
        created = self.store.create()
        self.token, self.state = created["session_id"], created["state"]

    def advance(self, action, **fields):
        self.clock.tick()
        self.state = self.store.advance(self.token, {"revision": self.state["revision"], "action": action, **fields})
        return self.state

    def option(self, role="correct"):
        item = self.items[self.state["question"]["item_id"]]
        return next(option["option_id"] for option in item["options"] if option["role"] == role)

    def assert_no_feedback(self, state):
        serialized = json.dumps(state, ensure_ascii=False)
        self.assertIsNone(state["profile"])
        for forbidden in ("correct_option_id", "explanation_zh", "selected_option_role", '"correct":'):
            self.assertNotIn(forbidden, serialized)
        for item in self.bank["items"]:
            self.assertNotIn(item["explanation_zh"], serialized)

    def finish(self, *, mixed=False):
        while self.state["phase"] != "completed":
            phase = self.state["phase"]
            if phase == "reading":
                self.assertIsNone(self.state["question"])
                self.advance("complete_reading")
            elif phase == "answering":
                role = "idk" if mixed and self.state["progress"]["answered"] == 0 else "correct"
                self.advance("answer", option_id=self.option(role))
            elif phase == "confidence":
                self.advance("confidence", value=None if mixed else 75)
            else:
                self.advance("review", value=True if mixed else False)
            if self.state["phase"] != "completed":
                self.assert_no_feedback(self.state)
        return self.state["profile"]

    def test_reading_payload_contains_no_question_or_target_metadata(self):
        self.assertEqual(self.state["phase"], "reading")
        self.assertEqual(self.state["progress"], {"answered": 0, "total": 10, "block_index": 1, "block_total": 4})
        serialized = json.dumps(self.state)
        self.assertIsNone(self.state["question"])
        for item in self.bank["items"]:
            self.assertNotIn(item["prompt"], serialized)
            self.assertNotIn(item["item_id"], serialized)
        with self.assertRaises(LearnerWorkspaceError) as caught:
            self.store.advance(self.token, {"revision": 0, "action": "answer", "option_id": "anything"})
        self.assertEqual(caught.exception.status, 409)

    def test_portable_record_is_unavailable_until_complete(self):
        self.assertIsNone(self.state.get("record"))
        self.advance("complete_reading")
        self.assertIsNone(self.state.get("record"))
        self.finish(mixed=True)
        record = self.state.get("record")
        self.assertIsInstance(record, dict)
        self.assertEqual(record["schema_version"], "lexigaze-reading-record-v1")
        self.assertEqual(record["profile"], self.state["profile"])
        self.assertFalse(record["research_evidence"])
        self.assertNotIn(self.token, json.dumps(record))
        self.assertNotIn("session_id", json.dumps(record))
        first = record["profile"]["observations"][0]
        self.assertTrue(first["is_idk"])
        self.assertFalse(first["correct"])
        self.assertTrue(first["selected_option_id"])
        self.assertEqual(first["confidence_reason"], "idk_not_applicable")

    def test_first_question_cutoff_is_fixed_before_later_blocks_and_outcomes(self):
        self.advance("complete_reading")
        self.finish()
        record = self.state.get("record")
        self.assertIsInstance(record, dict)
        measurement = record["measurement"]
        self.assertEqual(measurement["predictor_status"], "not_collected")
        self.assertEqual(measurement["first_question_cutoff_at"], measurement["reading_blocks"][0]["cutoff_at"])
        self.assertLess(measurement["first_question_cutoff_at"], measurement["reading_blocks"][1]["cutoff_at"])
        self.assertNotIn('"correct":', json.dumps(measurement))
        self.assertNotIn('"confidence":', json.dumps(measurement))
        self.assertEqual(len(record["instrument"]["content_sha256"]), 64)
        for observation in record["profile"]["observations"]:
            block = measurement["reading_blocks"][observation["block_index"] - 1]
            self.assertEqual(observation["reading_ended_at"], block["cutoff_at"])
            self.assertLessEqual(block["cutoff_at"], observation["answer_locked_at"])
            self.assertEqual(len(block["content_sha256"]), 64)

    def test_local_file_roundtrip_and_malformed_record_rejections(self):
        self.finish(mixed=True)
        result = subprocess.run(
            ["node", "scripts/test_learner_workspace_record.cjs"],
            input=json.dumps(self.state["record"], ensure_ascii=False),
            text=True, encoding="utf-8", capture_output=True, cwd=ROOT, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("record roundtrip and rejection checks passed", result.stdout)

    def test_only_current_question_is_available_after_reading_lock(self):
        self.advance("complete_reading")
        current = self.state["question"]
        item = self.items[current["item_id"]]
        idk = next(option for option in item["options"] if option["role"] == "idk")
        self.assertEqual(current["options"][-1]["option_id"], idk["option_id"])
        self.assertEqual(set(current), {"item_id", "prompt", "options"})
        self.assert_no_feedback(self.state)
        for other in self.bank["items"]:
            if other["item_id"] != item["item_id"]:
                self.assertNotIn(other["prompt"], json.dumps(self.state))

    def test_answer_then_confidence_then_review_are_locked_and_separate(self):
        self.advance("complete_reading")
        reading_revision = self.state["revision"]
        self.advance("answer", option_id=self.option())
        self.assertEqual(self.state["phase"], "confidence")
        self.assertEqual(self.state["progress"]["answered"], 1)
        self.assertIsNotNone(self.state["selected_text"])
        with self.assertRaises(LearnerWorkspaceError) as caught:
            self.store.advance(self.token, {"revision": reading_revision, "action": "complete_reading"})
        self.assertEqual(caught.exception.status, 409)
        with self.assertRaises(LearnerWorkspaceError) as caught:
            self.store.advance(self.token, {"revision": self.state["revision"], "action": "answer", "option_id": self.option("idk")})
        self.assertEqual(caught.exception.status, 409)
        self.advance("confidence", value=100)
        self.advance("review", value=None)
        profile = self.finish()
        observation = profile["observations"][0]
        self.assertTrue(observation["correct"])
        self.assertEqual(observation["confidence"], 100)
        self.assertEqual(observation["confidence_reason"], "provided")
        self.assertIsNone(observation["review_requested"])
        self.assertEqual(observation["review_reason"], "skipped")
        self.assertLess(observation["reading_started_at"], observation["reading_ended_at"])
        self.assertLess(observation["reading_ended_at"], observation["answer_locked_at"])
        self.assertLess(observation["answer_locked_at"], observation["confidence_observed_at"])
        self.assertEqual(observation["reading_ended_at"], profile["observations"][3]["reading_ended_at"])

    def test_idk_skips_confidence_and_optional_values_keep_missing_reasons(self):
        self.advance("complete_reading")
        self.advance("answer", option_id=self.option("idk"))
        self.assertEqual(self.state["phase"], "review")
        with self.assertRaises(LearnerWorkspaceError):
            self.advance("confidence", value=25)
        self.advance("review", value=True)
        profile = self.finish(mixed=True)
        first = profile["observations"][0]
        self.assertFalse(first["correct"])
        self.assertTrue(first["is_idk"])
        self.assertIsNone(first["confidence"])
        self.assertEqual(first["confidence_reason"], "idk_not_applicable")
        self.assertTrue(first["review_requested"])
        self.assertEqual(profile["observations"][1]["confidence_reason"], "skipped")
        self.assertEqual(sum(d["idk_count"] for d in profile["dimensions"]), 1)

    def test_four_blocks_complete_to_typed_profile_without_latent_score(self):
        profile = self.finish()
        self.assertEqual(self.state["progress"]["block_index"], 4)
        self.assertIsNone(self.state["context"])
        self.assertIsNone(self.state["question"])
        self.assertEqual(profile["scope"], "this_session_only")
        self.assertEqual(profile["status"], "developer_preview_observations")
        self.assertEqual([d["planned"] for d in profile["dimensions"]], [6, 2, 2])
        self.assertEqual([d["correct"] for d in profile["dimensions"]], [6, 2, 2])
        self.assertEqual(profile["reading"]["completed_blocks"], 4)
        self.assertEqual(profile["reading"]["elapsed_ms"], 4000)
        self.assertEqual(len(profile["observations"]), 10)
        self.assertEqual(profile["unavailable"], ["delayed_retention", "gaze", "cognitive_ability"])
        self.assertFalse(profile["research_evidence"])
        self.assertFalse(profile["persisted"])
        self.assertNotIn("total_score", profile)
        for item in profile["observations"]:
            self.assertEqual(item["instrument_id"], self.bank["walkthrough_id"])
            self.assertEqual(item["instrument_status"], self.bank["status"])
            self.assertEqual(item["source"], "learner_workspace_v1/objective_option_response")
            self.assertTrue(item["correct_text"])
            self.assertTrue(item["explanation_zh"])
        with self.assertRaises(LearnerWorkspaceError) as caught:
            self.advance("complete_reading")
        self.assertEqual(caught.exception.status, 409)

    def test_high_confidence_wrong_answer_does_not_change_correctness(self):
        self.advance("complete_reading")
        item = self.items[self.state["question"]["item_id"]]
        wrong = next(option for option in item["options"] if option["role"] not in {"correct", "idk"})
        self.advance("answer", option_id=wrong["option_id"])
        self.advance("confidence", value=100)
        self.advance("review", value=False)
        observation = self.finish()["observations"][0]
        self.assertFalse(observation["correct"])
        self.assertFalse(observation["is_idk"])
        self.assertEqual(observation["confidence"], 100)
        self.assertFalse(observation["review_requested"])

    def test_backward_clock_does_not_rewrite_reading_or_answer_state(self):
        self.advance("complete_reading")
        prior = self.state
        self.clock.tick(-2)
        with self.assertRaises(LearnerWorkspaceError) as caught:
            self.store.advance(self.token, {"revision": prior["revision"], "action": "answer", "option_id": self.option()})
        self.assertEqual(caught.exception.status, 503)
        self.clock.tick(2)
        self.assertEqual(self.store.get_state(self.token), prior)

    def test_sessions_and_returned_objects_are_isolated(self):
        second = self.store.create()
        self.advance("complete_reading")
        self.state["question"]["options"].clear()
        self.assertTrue(self.store.get_state(self.token)["question"]["options"])
        self.assertEqual(self.store.get_state(second["session_id"])["phase"], "reading")
        for token in (None, "foreign", "", []):
            with self.assertRaises(LearnerWorkspaceError) as caught:
                self.store.get_state(token)
            self.assertEqual(caught.exception.status, 404)
        self.store.delete(self.token)
        with self.assertRaises(LearnerWorkspaceError):
            self.store.get_state(self.token)
        self.assertEqual(self.store.get_state(second["session_id"])["revision"], 0)

    def test_expiry_is_fixed_and_capacity_recovers_after_expiry_or_delete(self):
        store = LearnerWorkspaceStore(clock=self.clock, max_sessions=1)
        first = store.create()
        with self.assertRaises(LearnerWorkspaceError) as caught:
            store.create()
        self.assertEqual(caught.exception.status, 429)
        self.clock.tick(SESSION_TTL_SECONDS - 1)
        self.assertEqual(store.get_state(first["session_id"])["expires_at"], first["state"]["expires_at"])
        self.clock.tick()
        with self.assertRaises(LearnerWorkspaceError):
            store.get_state(first["session_id"])
        second = store.create()
        store.delete(second["session_id"])
        self.assertTrue(store.create()["session_id"])

    def test_invalid_values_and_extra_fields_do_not_advance_or_change_state(self):
        invalid = [None, {}, {"revision": True, "action": "complete_reading"},
                   {"revision": 0, "action": []},
                   {"revision": 0, "action": "complete_reading", "reading_ended_at": "forged"}]
        for payload in invalid:
            with self.assertRaises(LearnerWorkspaceError) as caught:
                self.store.advance(self.token, payload)
            self.assertEqual(caught.exception.status, 400)
            self.assertEqual(self.store.get_state(self.token), self.state)
        self.advance("complete_reading")
        self.advance("answer", option_id=self.option())
        for value in (True, 0, 75.0, "75", [], {}):
            with self.assertRaises(LearnerWorkspaceError):
                self.advance("confidence", value=value)
        self.advance("confidence", value=None)
        for value in (0, 1, "true", [], {}):
            with self.assertRaises(LearnerWorkspaceError):
                self.advance("review", value=value)
        self.assertEqual(self.store.get_state(self.token), self.state)

    def test_concurrent_duplicate_transition_has_only_one_winner(self):
        barrier = threading.Barrier(2)
        def attempt():
            barrier.wait()
            try:
                return self.store.advance(self.token, {"revision": 0, "action": "complete_reading"})["revision"]
            except LearnerWorkspaceError as exc:
                return exc.status
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: attempt(), range(2)))
        self.assertEqual(sorted(results), [1, 409])


class WorkspaceSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.app = create_learner_workspace_app({"TESTING": True, "LEXIGAZE_LEARNER_CLOCK": self.clock})
        self.client = self.app.test_client()

    def request(self, method, path, **kwargs):
        return self.client.open(path, method=method, base_url=BASE_URL, **kwargs)

    def create(self):
        response = self.request("POST", f"{API}/session", json={})
        self.assertEqual(response.status_code, 200)
        return response.get_json()

    def test_api_lifecycle_and_request_headers(self):
        created = self.create()
        headers = {"X-Learner-Session": created["session_id"]}
        response = self.request("GET", f"{API}/state", headers=headers)
        self.assertEqual(response.get_json()["state"], created["state"])
        self.assertEqual(response.headers["Cache-Control"], "no-store, max-age=0")
        self.assertIn("camera=()", response.headers["Permissions-Policy"])
        self.assertNotIn("Set-Cookie", response.headers)
        response = self.request("POST", f"{API}/advance", headers=headers, json={"revision": 0, "action": "complete_reading"})
        self.assertEqual(response.get_json()["state"]["phase"], "answering")
        stale = self.request("POST", f"{API}/advance", headers=headers, json={"revision": 0, "action": "complete_reading"})
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(self.request("DELETE", f"{API}/session", headers=headers).get_json(), {"ok": True})
        self.assertEqual(self.request("GET", f"{API}/state", headers=headers).status_code, 404)

    def test_all_ten_answers_end_to_end_with_no_early_feedback(self):
        created = self.create()
        headers = {"X-Learner-Session": created["session_id"]}
        state = created["state"]
        items = {item["item_id"]: item for item in load_walkthrough_bank()["items"]}
        while state["phase"] != "completed":
            payload = {"revision": state["revision"]}
            if state["phase"] == "reading":
                self.assertIsNone(state["question"])
                payload["action"] = "complete_reading"
            elif state["phase"] == "answering":
                payload.update(action="answer", option_id=items[state["question"]["item_id"]]["correct_option_id"])
            elif state["phase"] == "confidence":
                payload.update(action="confidence", value=75)
            else:
                payload.update(action="review", value=False)
            response = self.request("POST", f"{API}/advance", headers=headers, json=payload)
            self.assertEqual(response.status_code, 200)
            state = response.get_json()["state"]
            if state["phase"] != "completed":
                self.assertIsNone(state["profile"])
                self.assertNotIn("correct_text", json.dumps(state))
        self.assertEqual(len(state["profile"]["observations"]), 10)

    def test_surface_rejects_other_hosts_origins_routes_methods_and_queries(self):
        cases = [
            ("GET", f"{API}/health?debug=1", {}, 400),
            ("GET", "/api/sessions", {}, 404),
            ("POST", f"{API}/health", {}, 405),
            ("GET", f"{API}/health", {"headers": {"Origin": "http://evil.test"}}, 403),
            ("GET", f"{API}/health", {"headers": {"Sec-Fetch-Site": "cross-site"}}, 403),
            ("GET", f"{API}/health", {"environ_overrides": {"REMOTE_ADDR": "192.0.2.1"}}, 403),
            ("GET", f"{API}/state", {}, 404),
        ]
        for method, path, kwargs, expected in cases:
            response = self.request(method, path, **kwargs)
            self.assertEqual(response.status_code, expected, path)
            self.assertIn("no-store", response.headers["Cache-Control"])
        response = self.client.get(f"{API}/health", base_url="http://localhost:8097")
        self.assertEqual(response.status_code, 421)
        for authority in ("0.0.0.0:8097", "localhost:8097", "127.0.0.1:80"):
            with self.assertRaises(ValueError):
                create_learner_workspace_app({"LEXIGAZE_LEARNER_AUTHORITY": authority})

    def test_bad_payloads_and_large_requests_are_rejected(self):
        for payload in ([], None, {"identity": "not accepted"}):
            response = self.request("POST", f"{API}/session", data=json.dumps(payload), content_type="application/json")
            self.assertEqual(response.status_code, 400)
        self.assertEqual(self.request("POST", f"{API}/session", data="{}").status_code, 415)
        response = self.request("POST", f"{API}/session", data='{"padding":"' + "x" * 65536 + '"}', content_type="application/json")
        self.assertEqual(response.status_code, 413)

    def test_api_operations_do_not_write_to_disk(self):
        with patch("builtins.open", side_effect=AssertionError("unexpected file access")), patch.object(Path, "open", side_effect=AssertionError("unexpected path access")):
            created = self.create()
            headers = {"X-Learner-Session": created["session_id"]}
            self.assertEqual(self.request("GET", f"{API}/state", headers=headers).status_code, 200)
            self.assertEqual(self.request("POST", f"{API}/advance", headers=headers, json={"revision": 0, "action": "complete_reading"}).status_code, 200)
            self.assertEqual(self.request("DELETE", f"{API}/session", headers=headers).status_code, 200)

    def test_fresh_import_does_not_load_participant_or_heavy_model_modules(self):
        script = (
            "import sys; from web.learner_workspace_app import create_learner_workspace_app; "
            "create_learner_workspace_app(); "
            "assert not any(n == 'torch' or n == 'cv2' or n.startswith('core.participant_study') for n in sys.modules)"
        )
        result = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", script], cwd=ROOT, text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
