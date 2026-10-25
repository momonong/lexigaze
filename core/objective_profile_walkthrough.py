"""Validated item bank and scoring for the objective-profile self walkthrough.

This module intentionally implements a bounded developer preview, not a
psychometric instrument.  It has no persistence and exposes no model score.
"""

from __future__ import annotations

import json
import random
import secrets
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK_PATH = ROOT / "core" / "objective_profile_walkthrough_v1.json"

WALKTHROUGH_ID = "lexigaze-objective-reading-walkthrough-v1"
WALKTHROUGH_STATUS = "developer_preview_only_unvalidated_items"
EXPECTED_KIND_COUNTS = {
    "contextual_meaning": 6,
    "passage_comprehension": 2,
    "novel_context_application": 2,
}
EXPECTED_CLAIM_BOUNDARY = {
    "allowed": [
        "研究者本人檢查流程、計分與結果呈現",
        "描述本次十題中實際觀察到的作答",
    ],
    "prohibited": [
        "題目或 learner profile 已具效度",
        "一般認知能力、注意力、記憶力、英文程度或 CEFR",
        "穩定的 learner type 或閱讀速度能力",
        "gaze 的效用、個人化協助效果、學習或延遲保留",
    ],
}
OPTION_ROLES = frozenset(
    {
        "correct",
        "competing_meaning",
        "local_detail_misread",
        "unsupported_inference",
        "opposite_or_reversed",
        "surface_association",
        "idk",
    }
)

PAIR_LABELS_ZH = {
    (True, True): "在兩個受測情境中皆答對",
    (True, False): "原情境答對；新情境未觀察到正確作答",
    (False, True): "兩個情境結果不一致",
    (False, False): "兩個受測情境都未觀察到正確作答",
}


class WalkthroughContractError(ValueError):
    """Raised when the preview bank or a score request violates its contract."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WalkthroughContractError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise WalkthroughContractError(f"{label} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise WalkthroughContractError(f"{label} fields are invalid")


def validate_walkthrough_bank(bank: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless a bank satisfies the frozen developer-preview shape."""

    bank = _mapping(bank, "bank")
    _exact_keys(
        bank,
        {
            "schema_version",
            "walkthrough_id",
            "status",
            "language",
            "title_zh",
            "intro_zh",
            "target_duration_minutes",
            "claim_boundary",
            "passages",
            "items",
        },
        "bank",
    )
    if bank["schema_version"] != 1:
        raise WalkthroughContractError("schema_version must be 1")
    if bank["walkthrough_id"] != WALKTHROUGH_ID:
        raise WalkthroughContractError("walkthrough_id is invalid")
    if bank["status"] != WALKTHROUGH_STATUS:
        raise WalkthroughContractError("walkthrough status is invalid")
    if bank["language"] != "en":
        raise WalkthroughContractError("walkthrough language must be en")
    for field in ("title_zh", "intro_zh"):
        if not isinstance(bank[field], str) or not bank[field].strip():
            raise WalkthroughContractError(f"{field} must be non-empty")

    duration = _mapping(bank["target_duration_minutes"], "target duration")
    _exact_keys(duration, {"minimum", "maximum"}, "target duration")
    if duration != {"minimum": 10, "maximum": 15}:
        raise WalkthroughContractError("target duration must remain 10-15 minutes")

    boundary = _mapping(bank["claim_boundary"], "claim boundary")
    _exact_keys(boundary, {"allowed", "prohibited"}, "claim boundary")
    for key in ("allowed", "prohibited"):
        values = _sequence(boundary[key], f"claim boundary {key}")
        if not values or any(not isinstance(value, str) or not value for value in values):
            raise WalkthroughContractError(f"claim boundary {key} is invalid")
    if dict(boundary) != EXPECTED_CLAIM_BOUNDARY:
        raise WalkthroughContractError("claim boundary cannot be relaxed or rewritten")

    passages = _sequence(bank["passages"], "passages")
    if len(passages) != 2:
        raise WalkthroughContractError("walkthrough must contain exactly 2 passages")
    passage_ids: set[str] = set()
    passage_item_ids: set[str] = set()
    for index, raw_passage in enumerate(passages):
        passage = _mapping(raw_passage, f"passage {index}")
        _exact_keys(
            passage,
            {"passage_id", "title", "text", "item_ids"},
            f"passage {index}",
        )
        passage_id = passage["passage_id"]
        if not isinstance(passage_id, str) or not passage_id or passage_id in passage_ids:
            raise WalkthroughContractError("passage IDs must be unique strings")
        passage_ids.add(passage_id)
        if not isinstance(passage["title"], str) or not passage["title"].strip():
            raise WalkthroughContractError(f"passage {passage_id} title is invalid")
        if not isinstance(passage["text"], str):
            raise WalkthroughContractError(f"passage {passage_id} text is invalid")
        word_count = len(passage["text"].split())
        if not 120 <= word_count <= 160:
            raise WalkthroughContractError(
                f"passage {passage_id} must contain 120-160 whitespace-delimited words"
            )
        item_ids = _sequence(passage["item_ids"], f"passage {passage_id} item_ids")
        if len(item_ids) != 4 or any(not isinstance(item_id, str) for item_id in item_ids):
            raise WalkthroughContractError(
                f"passage {passage_id} must reference exactly 4 item IDs"
            )
        if passage_item_ids.intersection(item_ids):
            raise WalkthroughContractError("passage item IDs must be unique")
        passage_item_ids.update(item_ids)

    items = _sequence(bank["items"], "items")
    if len(items) != 10:
        raise WalkthroughContractError("walkthrough must contain exactly 10 items")
    item_ids: set[str] = set()
    kind_counts: Counter[str] = Counter()
    item_by_id: dict[str, Mapping[str, Any]] = {}
    contextual_by_target: dict[str, str] = {}
    transfer_pairs: set[str] = set()
    global_option_ids: set[str] = set()
    for index, raw_item in enumerate(items):
        item = _mapping(raw_item, f"item {index}")
        _exact_keys(
            item,
            {
                "item_id",
                "passage_id",
                "kind",
                "target",
                "context",
                "prompt",
                "options",
                "correct_option_id",
                "explanation_zh",
                "paired_contextual_item_id",
            },
            f"item {index}",
        )
        item_id = item["item_id"]
        if not isinstance(item_id, str) or not item_id or item_id in item_ids:
            raise WalkthroughContractError("item IDs must be unique strings")
        item_ids.add(item_id)
        item_by_id[item_id] = item
        kind = item["kind"]
        if kind not in EXPECTED_KIND_COUNTS:
            raise WalkthroughContractError(f"item {item_id} kind is invalid")
        kind_counts[kind] += 1
        passage_id = item["passage_id"]
        if passage_id not in passage_ids:
            raise WalkthroughContractError(f"item {item_id} passage_id is invalid")
        for field in ("prompt", "explanation_zh"):
            if not isinstance(item[field], str) or not item[field].strip():
                raise WalkthroughContractError(f"item {item_id} {field} is invalid")

        target = item["target"]
        context = item["context"]
        pair_id = item["paired_contextual_item_id"]
        if kind == "passage_comprehension":
            if target is not None or context is not None or pair_id is not None:
                raise WalkthroughContractError(
                    f"comprehension item {item_id} has forbidden target fields"
                )
        elif kind == "contextual_meaning":
            if not isinstance(target, str) or not target or context is not None or pair_id is not None:
                raise WalkthroughContractError(
                    f"contextual item {item_id} target fields are invalid"
                )
            if target in contextual_by_target:
                raise WalkthroughContractError("contextual targets must be unique")
            contextual_by_target[target] = item_id
        else:
            if (
                not isinstance(target, str)
                or not target
                or not isinstance(context, str)
                or not context.strip()
                or not isinstance(pair_id, str)
            ):
                raise WalkthroughContractError(
                    f"transfer item {item_id} target fields are invalid"
                )
            transfer_pairs.add(pair_id)

        options = _sequence(item["options"], f"item {item_id} options")
        if len(options) != 5:
            raise WalkthroughContractError(f"item {item_id} must have 4 options plus IDK")
        option_ids: set[str] = set()
        role_counts: Counter[str] = Counter()
        for option_index, raw_option in enumerate(options):
            option = _mapping(raw_option, f"item {item_id} option {option_index}")
            _exact_keys(option, {"option_id", "text", "role"}, "option")
            option_id = option["option_id"]
            if not isinstance(option_id, str) or not option_id or option_id in option_ids:
                raise WalkthroughContractError(f"item {item_id} option IDs are invalid")
            option_ids.add(option_id)
            if option_id in global_option_ids:
                raise WalkthroughContractError("option IDs must be globally unique")
            global_option_ids.add(option_id)
            if not isinstance(option["text"], str) or not option["text"].strip():
                raise WalkthroughContractError(f"item {item_id} option text is invalid")
            role = option["role"]
            if role not in OPTION_ROLES:
                raise WalkthroughContractError(f"item {item_id} option role is invalid")
            role_counts[role] += 1
        if role_counts["correct"] != 1 or role_counts["idk"] != 1:
            raise WalkthroughContractError(
                f"item {item_id} needs exactly one correct and one IDK option"
            )
        if item["correct_option_id"] not in option_ids:
            raise WalkthroughContractError(f"item {item_id} correct option is unavailable")
        correct_option = next(
            option for option in options if option["option_id"] == item["correct_option_id"]
        )
        if correct_option["role"] != "correct":
            raise WalkthroughContractError(f"item {item_id} key is not the correct role")

    if dict(kind_counts) != EXPECTED_KIND_COUNTS:
        raise WalkthroughContractError("item-kind distribution is invalid")
    non_transfer_ids = {
        item_id
        for item_id, item in item_by_id.items()
        if item["kind"] != "novel_context_application"
    }
    if passage_item_ids != non_transfer_ids:
        raise WalkthroughContractError("passage item references do not match non-transfer items")
    for raw_passage in passages:
        passage = _mapping(raw_passage, "passage")
        referenced_items = [item_by_id[item_id] for item_id in passage["item_ids"]]
        if any(item["passage_id"] != passage["passage_id"] for item in referenced_items):
            raise WalkthroughContractError(
                f"passage {passage['passage_id']} references an item from another passage"
            )
        referenced_kinds = Counter(item["kind"] for item in referenced_items)
        if referenced_kinds != Counter(
            {"contextual_meaning": 3, "passage_comprehension": 1}
        ):
            raise WalkthroughContractError(
                f"passage {passage['passage_id']} must contain 3 contextual and 1 comprehension item"
            )
    if not transfer_pairs.issubset(set(contextual_by_target.values())):
        raise WalkthroughContractError("transfer pairs must reference contextual items")
    if len(transfer_pairs) != 2:
        raise WalkthroughContractError("exactly two contextual items must have transfer pairs")
    for item in items:
        if item["kind"] == "novel_context_application":
            source = item_by_id[item["paired_contextual_item_id"]]
            if (
                source["kind"] != "contextual_meaning"
                or source["target"] != item["target"]
                or source["passage_id"] != item["passage_id"]
            ):
                raise WalkthroughContractError("transfer target does not match its source item")

    return deepcopy(dict(bank))


def load_walkthrough_bank(path: Path | str = DEFAULT_BANK_PATH) -> dict[str, Any]:
    """Load and validate the versioned preview bank."""

    source = Path(path)
    try:
        parsed = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WalkthroughContractError(f"unable to load walkthrough bank: {exc}") from exc
    return validate_walkthrough_bank(parsed)


def public_walkthrough_form(
    bank: Mapping[str, Any],
    *,
    rng: random.Random | secrets.SystemRandom | None = None,
) -> dict[str, Any]:
    """Return a browser-safe form with keys, roles, and explanations removed."""

    bank = validate_walkthrough_bank(bank)
    shuffler = rng or secrets.SystemRandom()
    public_items: list[dict[str, Any]] = []
    for item in bank["items"]:
        options = [
            {"option_id": option["option_id"], "text": option["text"]}
            for option in item["options"]
            if option["role"] != "idk"
        ]
        shuffler.shuffle(options)
        idk_option = next(option for option in item["options"] if option["role"] == "idk")
        options.append(
            {"option_id": idk_option["option_id"], "text": idk_option["text"]}
        )
        public_items.append(
            {
                "item_id": item["item_id"],
                "passage_id": item["passage_id"],
                "kind": item["kind"],
                "target": item["target"],
                "context": item["context"],
                "prompt": item["prompt"],
                "options": options,
            }
        )
    return {
        "schema_version": bank["schema_version"],
        "walkthrough_id": bank["walkthrough_id"],
        "status": bank["status"],
        "language": bank["language"],
        "title_zh": bank["title_zh"],
        "intro_zh": bank["intro_zh"],
        "target_duration_minutes": deepcopy(bank["target_duration_minutes"]),
        "claim_boundary": deepcopy(bank["claim_boundary"]),
        "passages": deepcopy(bank["passages"]),
        "items": public_items,
        "storage": "request_memory_only_reload_clears_answers",
        "camera": "disabled",
        "gpu": "not_applicable_no_model_inference",
    }


def _validated_elapsed_ms(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WalkthroughContractError("total_elapsed_ms must be an integer")
    if not 0 <= value <= 2 * 60 * 60 * 1000:
        raise WalkthroughContractError("total_elapsed_ms is outside the accepted range")
    return value


def score_walkthrough(
    bank: Mapping[str, Any],
    responses: Mapping[str, Any],
    *,
    total_elapsed_ms: Any,
) -> dict[str, Any]:
    """Score a complete response set without fitting or persisting a model."""

    bank = validate_walkthrough_bank(bank)
    responses = _mapping(responses, "responses")
    elapsed_ms = _validated_elapsed_ms(total_elapsed_ms)
    item_by_id = {item["item_id"]: item for item in bank["items"]}
    if set(responses) != set(item_by_id):
        raise WalkthroughContractError("responses must contain every planned item exactly once")

    dimensions = {
        kind: {
            "correct": 0,
            "incorrect": 0,
            "answered": 0,
            "planned": planned,
            "idk": 0,
            "missing": 0,
        }
        for kind, planned in EXPECTED_KIND_COUNTS.items()
    }
    error_patterns: Counter[str] = Counter()
    item_results: list[dict[str, Any]] = []
    correctness: dict[str, bool] = {}
    for item in bank["items"]:
        item_id = item["item_id"]
        selected_id = responses[item_id]
        if not isinstance(selected_id, str):
            raise WalkthroughContractError(f"response for {item_id} must be an option ID")
        option_by_id = {option["option_id"]: option for option in item["options"]}
        selected = option_by_id.get(selected_id)
        if selected is None:
            raise WalkthroughContractError(f"response for {item_id} is invalid")
        correct_option = option_by_id[item["correct_option_id"]]
        is_correct = selected_id == item["correct_option_id"]
        correctness[item_id] = is_correct
        dimension = dimensions[item["kind"]]
        dimension["answered"] += 1
        dimension["correct"] += int(is_correct)
        dimension["incorrect"] += int(not is_correct)
        if selected["role"] == "idk":
            dimension["idk"] += 1
        elif not is_correct:
            error_patterns[selected["role"]] += 1
        item_results.append(
            {
                "item_id": item_id,
                "kind": item["kind"],
                "target": item["target"],
                "prompt": item["prompt"],
                "selected_option_id": selected_id,
                "selected_text": selected["text"],
                "correct_option_id": item["correct_option_id"],
                "correct_text": correct_option["text"],
                "correct": is_correct,
                "selected_option_role": selected["role"],
                "explanation_zh": item["explanation_zh"],
            }
        )

    paired_evidence: list[dict[str, Any]] = []
    paired_source_ids: set[str] = set()
    for item in bank["items"]:
        if item["kind"] != "novel_context_application":
            continue
        source_id = item["paired_contextual_item_id"]
        paired_source_ids.add(source_id)
        pattern = (correctness[source_id], correctness[item["item_id"]])
        paired_evidence.append(
            {
                "target": item["target"],
                "source_item_id": source_id,
                "transfer_item_id": item["item_id"],
                "source_correct": pattern[0],
                "transfer_correct": pattern[1],
                "interpretation_zh": PAIR_LABELS_ZH[pattern],
            }
        )
    single_context_targets = [
        {
            "target": item["target"],
            "item_id": item["item_id"],
            "correct": correctness[item["item_id"]],
            "interpretation_zh": "只有單一受測情境的作答證據",
        }
        for item in bank["items"]
        if item["kind"] == "contextual_meaning"
        and item["item_id"] not in paired_source_ids
    ]

    total_idk = sum(value["idk"] for value in dimensions.values())
    return {
        "ok": True,
        "walkthrough_id": bank["walkthrough_id"],
        "status": "complete_software_preview",
        "summary_title_zh": "本次英語閱讀作答摘要",
        "dimensions": dimensions,
        "idk_count": total_idk,
        "missing_count": 0,
        "total_elapsed_ms": elapsed_ms,
        "timing_interpretation": "display_only_not_used_for_ability_inference",
        "error_patterns": dict(sorted(error_patterns.items())),
        "error_pattern_interpretation": (
            "selected_option_categories_only_not_validated_psychological_misconceptions"
        ),
        "paired_context_evidence": paired_evidence,
        "single_context_evidence": single_context_targets,
        "item_results": item_results,
        "uncertainty": "single_session_unvalidated_items",
        "claim_status": "software_preview_only_not_research_evidence",
        "claim_boundary": deepcopy(bank["claim_boundary"]),
        "model_fitting_authorized": False,
        "profile_validity_claim_authorized": False,
        "cognitive_ability_claim_authorized": False,
        "english_proficiency_claim_authorized": False,
        "gaze_claim_authorized": False,
        "learning_effect_claim_authorized": False,
        "persisted": False,
    }


__all__ = [
    "DEFAULT_BANK_PATH",
    "EXPECTED_CLAIM_BOUNDARY",
    "EXPECTED_KIND_COUNTS",
    "WALKTHROUGH_ID",
    "WALKTHROUGH_STATUS",
    "WalkthroughContractError",
    "load_walkthrough_bank",
    "public_walkthrough_form",
    "score_walkthrough",
    "validate_walkthrough_bank",
]
