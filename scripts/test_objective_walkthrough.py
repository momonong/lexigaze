"""Focused tests for the no-store objective reading walkthrough."""

from __future__ import annotations

import hashlib
import json
import random
import sys
import unittest
from pathlib import Path

MODULES_BEFORE_WALKTHROUGH_IMPORT = set(sys.modules)

from core.objective_profile_walkthrough import (
    EXPECTED_KIND_COUNTS,
    WalkthroughContractError,
    load_walkthrough_bank,
    public_walkthrough_form,
    score_walkthrough,
    validate_walkthrough_bank,
)
from web.objective_walkthrough_app import create_objective_walkthrough_app

MODULES_IMPORTED_BY_WALKTHROUGH = set(sys.modules) - MODULES_BEFORE_WALKTHROUGH_IMPORT


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8098"


def _responses(bank: dict, *, mode: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in bank["items"]:
        if mode == "correct":
            selected = item["correct_option_id"]
        elif mode == "idk":
            selected = next(
                option["option_id"]
                for option in item["options"]
                if option["role"] == "idk"
            )
        else:
            raise AssertionError(f"unknown fixture mode: {mode}")
        result[item["item_id"]] = selected
    return result


def _file_snapshot(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class WalkthroughBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bank = load_walkthrough_bank()

    def test_bank_has_two_passages_and_exact_ten_item_contract(self) -> None:
        self.assertEqual(len(self.bank["passages"]), 2)
        self.assertEqual(len(self.bank["items"]), 10)
        counts = {
            kind: sum(item["kind"] == kind for item in self.bank["items"])
            for kind in EXPECTED_KIND_COUNTS
        }
        self.assertEqual(counts, EXPECTED_KIND_COUNTS)
        self.assertEqual(
            [len(passage["text"].split()) for passage in self.bank["passages"]],
            [138, 149],
        )

    def test_public_form_hides_every_key_role_and_explanation(self) -> None:
        public = public_walkthrough_form(self.bank, rng=random.Random(7))
        serialized = json.dumps(public, ensure_ascii=False)
        for forbidden in (
            "correct_option_id",
            "selected_option_role",
            "paired_contextual_item_id",
            "explanation_zh",
            '"role"',
        ):
            self.assertNotIn(forbidden, serialized)
        for item in public["items"]:
            self.assertEqual(item["options"][-1]["text"], "I do not know")
            self.assertEqual(len(item["options"]), 5)

    def test_bank_validation_fails_closed_on_key_or_status_change(self) -> None:
        tampered = json.loads(json.dumps(self.bank))
        tampered["status"] = "collection_ready"
        with self.assertRaises(WalkthroughContractError):
            validate_walkthrough_bank(tampered)

        tampered = json.loads(json.dumps(self.bank))
        tampered["claim_boundary"] = {
            "allowed": ["可推論一般認知能力"],
            "prohibited": ["foo"],
        }
        with self.assertRaises(WalkthroughContractError):
            validate_walkthrough_bank(tampered)

        tampered = json.loads(json.dumps(self.bank))
        tampered["items"][0]["correct_option_id"] = "missing-option"
        with self.assertRaises(WalkthroughContractError):
            validate_walkthrough_bank(tampered)

    def test_bank_rejects_cross_passage_item_swaps_and_transfer_pairs(self) -> None:
        tampered = json.loads(json.dumps(self.bank))
        first_items = tampered["passages"][0]["item_ids"]
        second_items = tampered["passages"][1]["item_ids"]
        first_items[-1], second_items[-1] = second_items[-1], first_items[-1]
        with self.assertRaises(WalkthroughContractError):
            validate_walkthrough_bank(tampered)

        tampered = json.loads(json.dumps(self.bank))
        transfer = next(
            item
            for item in tampered["items"]
            if item["kind"] == "novel_context_application"
        )
        transfer["passage_id"] = "p2-island-navigation"
        with self.assertRaises(WalkthroughContractError):
            validate_walkthrough_bank(tampered)

    def test_browser_option_ids_are_opaque_and_keys_have_no_shared_suffix(self) -> None:
        public = public_walkthrough_form(self.bank, rng=random.Random(11))
        browser_ids = {
            option["option_id"]
            for item in public["items"]
            for option in item["options"]
            if option["text"] != "I do not know"
        }
        self.assertTrue(all(option_id.startswith("opt-") for option_id in browser_ids))
        self.assertTrue(all(len(option_id) == 12 for option_id in browser_ids))
        key_suffixes = {
            item["correct_option_id"].removeprefix("opt-") for item in self.bank["items"]
        }
        self.assertEqual(len(key_suffixes), len(self.bank["items"]))

    def test_scoring_keeps_three_domains_separate_and_has_no_total_score(self) -> None:
        result = score_walkthrough(
            self.bank,
            _responses(self.bank, mode="correct"),
            total_elapsed_ms=612_345,
        )
        self.assertEqual(
            {kind: value["correct"] for kind, value in result["dimensions"].items()},
            EXPECTED_KIND_COUNTS,
        )
        self.assertNotIn("total_score", result)
        self.assertNotIn("theta", result)
        self.assertEqual(result["idk_count"], 0)
        self.assertFalse(result["profile_validity_claim_authorized"])
        self.assertFalse(result["cognitive_ability_claim_authorized"])
        self.assertFalse(result["learning_effect_claim_authorized"])
        self.assertFalse(result["persisted"])

    def test_idk_is_scored_as_observed_noncorrect_response(self) -> None:
        result = score_walkthrough(
            self.bank,
            _responses(self.bank, mode="idk"),
            total_elapsed_ms=1000,
        )
        self.assertEqual(result["idk_count"], 10)
        self.assertEqual(result["missing_count"], 0)
        self.assertTrue(
            all(value["correct"] == 0 for value in result["dimensions"].values())
        )
        self.assertTrue(
            all(
                value["incorrect"] == value["planned"]
                and value["missing"] == 0
                for value in result["dimensions"].values()
            )
        )
        self.assertEqual(result["error_patterns"], {})

    def test_missing_unknown_and_malformed_responses_are_rejected(self) -> None:
        responses = _responses(self.bank, mode="correct")
        responses.pop(next(iter(responses)))
        with self.assertRaises(WalkthroughContractError):
            score_walkthrough(self.bank, responses, total_elapsed_ms=1000)
        responses = _responses(self.bank, mode="correct")
        responses[next(iter(responses))] = "unknown-option"
        with self.assertRaises(WalkthroughContractError):
            score_walkthrough(self.bank, responses, total_elapsed_ms=1000)
        with self.assertRaises(WalkthroughContractError):
            score_walkthrough(self.bank, [], total_elapsed_ms=1000)


class WalkthroughSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_objective_walkthrough_app({"TESTING": True})
        cls.client = cls.app.test_client()
        cls.bank = load_walkthrough_bank()

    def get(self, path: str, **kwargs):
        return self.client.get(path, base_url=BASE_URL, **kwargs)

    def post(self, path: str, **kwargs):
        return self.client.post(path, base_url=BASE_URL, **kwargs)

    def test_page_form_and_assets_have_no_store_camera_off_headers(self) -> None:
        for path in (
            "/objective-profile",
            "/api/objective-profile/form",
            "/objective-profile/assets/objective_walkthrough.css",
            "/objective-profile/assets/objective_walkthrough.js",
        ):
            response = self.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertEqual(response.headers["Cache-Control"], "no-store, max-age=0")
            self.assertIn("camera=()", response.headers["Permissions-Policy"])
            self.assertNotIn("Set-Cookie", response.headers)

    def test_health_states_bounded_runtime(self) -> None:
        response = self.get("/api/objective-profile/health")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["storage"], "none")
        self.assertEqual(payload["camera"], "disabled")
        self.assertEqual(payload["model_inference"], "none")
        self.assertEqual(payload["gpu"], "not_applicable")
        self.assertFalse(payload["research_claim_authorized"])

    def test_score_endpoint_scores_complete_answers_and_writes_no_study_files(self) -> None:
        study_root = ROOT / "data" / "participant_studies"
        before = _file_snapshot(study_root)
        response = self.post(
            "/api/objective-profile/score",
            json={
                "responses": _responses(self.bank, mode="correct"),
                "total_elapsed_ms": 555_000,
            },
        )
        after = _file_snapshot(study_root)
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertEqual(payload["claim_status"], "software_preview_only_not_research_evidence")
        self.assertFalse(payload["persisted"])
        self.assertEqual(before, after)

    def test_score_endpoint_rejects_incomplete_and_non_json_requests(self) -> None:
        response = self.post(
            "/api/objective-profile/score",
            json={"responses": {}, "total_elapsed_ms": 12},
        )
        self.assertEqual(response.status_code, 400)
        response = self.post(
            "/api/objective-profile/score",
            data="responses=no",
            content_type="application/x-www-form-urlencoded",
        )
        self.assertEqual(response.status_code, 415)

    def test_surface_rejects_wrong_host_remote_origin_query_route_and_method(self) -> None:
        wrong_host = self.client.get(
            "/objective-profile", base_url="http://localhost:8098"
        )
        self.assertEqual(wrong_host.status_code, 421)
        remote = self.get(
            "/objective-profile", environ_overrides={"REMOTE_ADDR": "192.0.2.10"}
        )
        self.assertEqual(remote.status_code, 403)
        cross_origin = self.get(
            "/api/objective-profile/form",
            headers={"Origin": "http://example.test"},
        )
        self.assertEqual(cross_origin.status_code, 403)
        self.assertEqual(self.get("/objective-profile?debug=1").status_code, 400)
        self.assertEqual(self.get("/api/sessions").status_code, 404)
        self.assertEqual(self.post("/objective-profile").status_code, 405)

    def test_client_source_uses_no_browser_storage_or_media_capture(self) -> None:
        source = self.get(
            "/objective-profile/assets/objective_walkthrough.js"
        ).get_data(as_text=True)
        for forbidden in (
            "localStorage",
            "sessionStorage",
            "indexedDB",
            "getUserMedia",
            "MediaRecorder",
            "navigator.mediaDevices",
        ):
            self.assertNotIn(forbidden, source)

    def test_dedicated_app_imports_no_participant_store_or_heavy_model_runtime(self) -> None:
        self.assertNotIn("torch", MODULES_IMPORTED_BY_WALKTHROUGH)
        self.assertNotIn("cv2", MODULES_IMPORTED_BY_WALKTHROUGH)
        self.assertFalse(
            any(
                name.startswith("core.participant_study")
                for name in MODULES_IMPORTED_BY_WALKTHROUGH
            )
        )

    def test_privileged_or_default_http_port_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            create_objective_walkthrough_app(
                {"LEXIGAZE_WALKTHROUGH_AUTHORITY": "127.0.0.1:80"}
            )


if __name__ == "__main__":
    unittest.main()
