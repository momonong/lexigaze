"""Ephemeral reading-to-observation flow for the owner's product preview.

The existing, unvalidated walkthrough bank is example content only. This module
does not fit a learner model, persist records, or establish research validity.
"""

from __future__ import annotations

import hashlib
import json
import math
import secrets
import threading
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from core.objective_profile_walkthrough import (
    load_walkthrough_bank,
    public_walkthrough_form,
    score_walkthrough,
    validate_walkthrough_bank,
)

SESSION_TTL_SECONDS = 30 * 60
DEFAULT_MAX_SESSIONS = 32
DIMENSION_LABELS = {
    "contextual_meaning": "語境字義",
    "passage_comprehension": "文章理解",
    "novel_context_application": "換情境作答",
}


class LearnerWorkspaceError(ValueError):
    """A bounded, public request rejection."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="milliseconds")


class LearnerWorkspaceStore:
    """Thread-safe process memory with fixed expiry and revisioned transitions.

    Session tokens are browser-memory capabilities, never cookies or identities.
    Expired sessions are deleted on each operation; reads never extend expiry.
    """

    def __init__(
        self,
        *,
        bank: Mapping[str, Any] | None = None,
        clock: Callable[[], float] = time.time,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
    ) -> None:
        if isinstance(max_sessions, bool) or not isinstance(max_sessions, int) or max_sessions < 1:
            raise ValueError("max_sessions must be a positive integer")
        self._bank = validate_walkthrough_bank(bank) if bank is not None else load_walkthrough_bank()
        self._items = {item["item_id"]: item for item in self._bank["items"]}
        self._instrument_sha256 = hashlib.sha256(
            json.dumps(self._bank, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self._clock = clock
        self._max_sessions = max_sessions
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def _now(self) -> float:
        value = float(self._clock())
        if not math.isfinite(value):
            raise LearnerWorkspaceError("session clock is unavailable", 503)
        return value

    def _cleanup(self, now: float) -> None:
        for token in [key for key, session in self._sessions.items() if now >= session["expires"]]:
            del self._sessions[token]

    def _session(self, token: object, now: float) -> dict[str, Any]:
        self._cleanup(now)
        if not isinstance(token, str) or token not in self._sessions:
            raise LearnerWorkspaceError("session is missing, expired, or unavailable", 404)
        session = self._sessions[token]
        if now < session["last_observed"]:
            raise LearnerWorkspaceError("session clock moved backwards; restart this preview", 503)
        return session

    def _blocks(self) -> list[dict[str, Any]]:
        blocks = [
            {
                "title": passage["title"],
                "text": passage["text"],
                "kind": "passage",
                "item_ids": list(passage["item_ids"]),
            }
            for passage in self._bank["passages"]
        ]
        for item in self._bank["items"]:
            if item["kind"] == "novel_context_application":
                blocks.append(
                    {
                        "title": f"換情境閱讀 {len(blocks) - 1}",
                        "text": item["context"],
                        "kind": "novel_context",
                        "item_ids": [item["item_id"]],
                    }
                )
        return blocks

    def create(self) -> dict[str, Any]:
        with self._lock:
            now = self._now()
            self._cleanup(now)
            if len(self._sessions) >= self._max_sessions:
                raise LearnerWorkspaceError("preview session limit reached; close a session or wait for expiry", 429)
            public = public_walkthrough_form(self._bank)
            token = secrets.token_urlsafe(32)
            while token in self._sessions:
                token = secrets.token_urlsafe(32)
            session: dict[str, Any] = {
                "created": now,
                "expires": now + SESSION_TTL_SECONDS,
                "last_observed": now,
                "revision": 0,
                "phase": "reading",
                "blocks": self._blocks(),
                "block_index": 0,
                "item_index": 0,
                "questions": {
                    item["item_id"]: {
                        "item_id": item["item_id"],
                        "prompt": item["prompt"],
                        "options": item["options"],
                    }
                    for item in public["items"]
                },
                "reading_started": now,
                "reading_ended": None,
                "reading_records": [],
                "first_question_cutoff": None,
                "responses": {},
                "observations": [],
                "pending": None,
                "profile": None,
                "record": None,
            }
            self._sessions[token] = session
            return {"session_id": token, "state": self._public_state(session)}

    def get_state(self, token: object) -> dict[str, Any]:
        with self._lock:
            return self._public_state(self._session(token, self._now()))

    def delete(self, token: object) -> None:
        with self._lock:
            self._session(token, self._now())
            del self._sessions[token]

    def _current_item(self, session: dict[str, Any]) -> dict[str, Any]:
        block = session["blocks"][session["block_index"]]
        return self._items[block["item_ids"][session["item_index"]]]

    def _public_state(self, session: dict[str, Any]) -> dict[str, Any]:
        completed = session["phase"] == "completed"
        block = session["blocks"][session["block_index"]]
        question = None
        if session["phase"] in {"answering", "confidence", "review"}:
            question = session["questions"][self._current_item(session)["item_id"]]
        return deepcopy(
            {
                "revision": session["revision"],
                "phase": session["phase"],
                "progress": {
                    "answered": len(session["responses"]),
                    "total": len(self._items),
                    "block_index": session["block_index"] + 1,
                    "block_total": len(session["blocks"]),
                },
                "context": None if completed else {key: block[key] for key in ("title", "text", "kind")},
                "question": question,
                "selected_text": session["pending"]["selected_text"] if session["pending"] else None,
                "profile": session["profile"],
                "record": session["record"],
                "expires_at": _timestamp(session["expires"]),
            }
        )

    def advance(self, token: object, payload: object) -> dict[str, Any]:
        with self._lock:
            now = self._now()
            session = self._session(token, now)
            if not isinstance(payload, Mapping):
                raise LearnerWorkspaceError("request must be an object")
            revision = payload.get("revision")
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
                raise LearnerWorkspaceError("revision must be a non-negative integer")
            if revision != session["revision"]:
                raise LearnerWorkspaceError("session changed; use its current state", 409)
            action = payload.get("action")
            expected_fields = {
                "complete_reading": {"revision", "action"},
                "answer": {"revision", "action", "option_id"},
                "confidence": {"revision", "action", "value"},
                "review": {"revision", "action", "value"},
            }
            if not isinstance(action, str) or action not in expected_fields or set(payload) != expected_fields[action]:
                raise LearnerWorkspaceError("action fields are invalid")
            required_phase = {
                "complete_reading": "reading", "answer": "answering",
                "confidence": "confidence", "review": "review",
            }[action]
            if session["phase"] != required_phase:
                raise LearnerWorkspaceError("action is unavailable at this step", 409)

            if action == "complete_reading":
                session["reading_ended"] = now
                if session["first_question_cutoff"] is None:
                    session["first_question_cutoff"] = now
                block = session["blocks"][session["block_index"]]
                session["reading_records"].append(
                    {
                        "started": session["reading_started"], "ended": now,
                        "block_index": session["block_index"] + 1,
                        "context_title": block["title"], "context_text": block["text"],
                        "content_sha256": hashlib.sha256(block["text"].encode("utf-8")).hexdigest(),
                    }
                )
                session["phase"] = "answering"
            elif action == "answer":
                self._answer(session, payload["option_id"], now)
            elif action == "confidence":
                value = payload["value"]
                if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value not in {25, 50, 75, 100}):
                    raise LearnerWorkspaceError("confidence must be 25, 50, 75, 100, or null")
                session["pending"]["confidence"] = value
                session["pending"]["confidence_reason"] = "skipped" if value is None else "provided"
                session["pending"]["confidence_observed_at"] = _timestamp(now)
                session["phase"] = "review"
            else:
                value = payload["value"]
                if value is not None and not isinstance(value, bool):
                    raise LearnerWorkspaceError("review preference must be boolean or null")
                self._review(session, value, now)
            session["last_observed"] = now
            session["revision"] += 1
            return self._public_state(session)

    def _answer(self, session: dict[str, Any], option_id: object, now: float) -> None:
        item = self._current_item(session)
        selected = next((option for option in item["options"] if option["option_id"] == option_id), None)
        if not isinstance(option_id, str) or selected is None:
            raise LearnerWorkspaceError("option is unavailable for this question")
        block = session["blocks"][session["block_index"]]
        idk = selected["role"] == "idk"
        session["responses"][item["item_id"]] = option_id
        session["pending"] = {
            "item_id": item["item_id"],
            "block_index": session["block_index"] + 1,
            "kind": item["kind"],
            "target": item["target"],
            "context_title": block["title"],
            "context_text": block["text"],
            "prompt": item["prompt"],
            "selected_text": selected["text"],
            "selected_option_id": option_id,
            "is_idk": idk,
            "confidence": None,
            "confidence_reason": "idk_not_applicable" if idk else "pending",
            "confidence_observed_at": None,
            "review_requested": None,
            "review_reason": "pending",
            "observed_at": _timestamp(now),
            "reading_started_at": _timestamp(session["reading_started"]),
            "reading_ended_at": _timestamp(session["reading_ended"]),
            "answer_locked_at": _timestamp(now),
            "instrument_id": self._bank["walkthrough_id"],
            "instrument_schema_version": self._bank["schema_version"],
            "instrument_status": self._bank["status"],
            "source": "learner_workspace_v1/objective_option_response",
            "workspace_version": "lexigaze-reading-learner-workspace-v1",
            "evidence_status": "unvalidated_example_item_observation",
        }
        session["phase"] = "review" if idk else "confidence"

    def _review(self, session: dict[str, Any], value: bool | None, now: float) -> None:
        observation = session["pending"]
        observation["review_requested"] = value
        observation["review_reason"] = "skipped" if value is None else "provided"
        observation["review_observed_at"] = _timestamp(now)
        session["observations"].append(observation)
        session["pending"] = None
        block = session["blocks"][session["block_index"]]
        if session["item_index"] + 1 < len(block["item_ids"]):
            session["item_index"] += 1
            session["phase"] = "answering"
        elif session["block_index"] + 1 < len(session["blocks"]):
            session["block_index"] += 1
            session["item_index"] = 0
            session["reading_started"] = now
            session["reading_ended"] = None
            session["phase"] = "reading"
        else:
            session["phase"] = "completed"
            session["profile"] = self._profile(session, now)
            session["record"] = self._record(session, now)

    def _record(self, session: dict[str, Any], now: float) -> dict[str, Any]:
        """Portable completed preview; no capability, identity or predictor payload."""
        return {
            "schema_version": "lexigaze-reading-record-v1",
            "kind": "completed_example_reading_record",
            "created_at": _timestamp(session["created"]),
            "completed_at": _timestamp(now),
            "instrument": {
                "id": self._bank["walkthrough_id"],
                "schema_version": self._bank["schema_version"],
                "status": self._bank["status"],
                "content_sha256": self._instrument_sha256,
                "hash_encoding": "utf8_sorted_compact_json",
            },
            "measurement": {
                "first_question_cutoff_at": _timestamp(session["first_question_cutoff"]),
                "predictor_status": "not_collected",
                "cutoff_scope": "timestamp_only_not_qualified_predictor_snapshot",
                "reading_blocks": [
                    {
                        "block_index": record["block_index"],
                        "started_at": _timestamp(record["started"]),
                        "cutoff_at": _timestamp(record["ended"]),
                        "context_title": record["context_title"],
                        "context_text": record["context_text"],
                        "content_sha256": record["content_sha256"],
                    }
                    for record in session["reading_records"]
                ],
            },
            "profile": deepcopy(session["profile"]),
            "research_evidence": False,
            "import_policy": "local_file_review_only_not_authenticated",
        }

    def _profile(self, session: dict[str, Any], now: float) -> dict[str, Any]:
        scored = score_walkthrough(
            self._bank, session["responses"], total_elapsed_ms=round((now - session["created"]) * 1000)
        )
        results = {item["item_id"]: item for item in scored["item_results"]}
        observations = []
        for observation in session["observations"]:
            result = results[observation["item_id"]]
            observations.append({**observation, **{key: result[key] for key in ("correct", "correct_text", "explanation_zh")}})
        return {
            "generated_at": _timestamp(now),
            "scope": "this_session_only",
            "status": "developer_preview_observations",
            "dimensions": [
                {
                    "key": kind, "label": DIMENSION_LABELS[kind],
                    "correct": counts["correct"], "answered": counts["answered"],
                    "planned": counts["planned"], "idk_count": counts["idk"],
                }
                for kind, counts in scored["dimensions"].items()
            ],
            "reading": {
                "completed_blocks": len(session["reading_records"]),
                "elapsed_ms": sum(round((record["ended"] - record["started"]) * 1000) for record in session["reading_records"]),
                "timing_interpretation": "server_observed_reading_phase_duration_not_attention_or_reading_speed",
            },
            "observations": observations,
            "unavailable": ["delayed_retention", "gaze", "cognitive_ability"],
            "uncertainty": "single_session_unvalidated_items",
            "research_evidence": False,
            "persisted": False,
        }


__all__ = ["LearnerWorkspaceError", "LearnerWorkspaceStore", "SESSION_TTL_SECONDS"]
