"""Routes for the memory-only objective reading walkthrough."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from flask import Blueprint, Response, current_app, jsonify, render_template, request

from core.objective_profile_walkthrough import (
    WalkthroughContractError,
    public_walkthrough_form,
    score_walkthrough,
)
from web.objective_walkthrough_surface import json_object_rejection


objective_walkthrough_bp = Blueprint("objective_walkthrough", __name__)

API_PREFIX = "/api/objective-profile"
BANK_EXTENSION = "lexigaze_objective_walkthrough_bank"
STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
ASSET_FILES = {
    "objective_walkthrough.css": STATIC_DIR / "objective_walkthrough.css",
    "objective_walkthrough.js": STATIC_DIR / "objective_walkthrough.js",
}


def _bank() -> Mapping[str, Any]:
    bank = current_app.extensions.get(BANK_EXTENSION)
    if not isinstance(bank, Mapping):
        raise RuntimeError("objective walkthrough bank is unavailable")
    return bank


def _error(status: int, detail: str) -> Response:
    response = jsonify(
        {
            "ok": False,
            "classification": "objective_walkthrough_request_rejected",
            "error": detail,
            "research_claim_authorized": False,
            "persisted": False,
        }
    )
    response.status_code = status
    return response


@objective_walkthrough_bp.get("/objective-profile")
def page() -> str:
    return render_template("objective_walkthrough.html")


@objective_walkthrough_bp.get("/objective-profile/assets/<filename>")
def asset(filename: str) -> Response:
    path = ASSET_FILES.get(filename)
    if path is None:
        return _error(404, "asset is unavailable")
    mimetype = "text/css" if path.suffix == ".css" else "text/javascript"
    return current_app.response_class(path.read_bytes(), mimetype=mimetype)


@objective_walkthrough_bp.get(f"{API_PREFIX}/health")
def health() -> Response:
    return jsonify(
        {
            "ok": True,
            "service": "objective-profile-self-walkthrough-v1",
            "scope": "researcher_software_preview_only",
            "loopback_only": True,
            "storage": "none",
            "camera": "disabled",
            "model_inference": "none",
            "gpu": "not_applicable",
            "research_claim_authorized": False,
        }
    )


@objective_walkthrough_bp.get(f"{API_PREFIX}/form")
def form() -> Response:
    return jsonify(public_walkthrough_form(_bank()))


@objective_walkthrough_bp.post(f"{API_PREFIX}/score")
def score() -> Response:
    parsed = request.get_json(silent=True)
    rejection = json_object_rejection(mimetype=request.mimetype, parsed_json=parsed)
    if rejection is not None:
        status, detail = rejection
        return _error(status, detail)
    body = dict(parsed)
    if set(body) != {"responses", "total_elapsed_ms"}:
        return _error(400, "request fields are invalid")
    try:
        result = score_walkthrough(
            _bank(),
            body["responses"],
            total_elapsed_ms=body["total_elapsed_ms"],
        )
    except WalkthroughContractError as exc:
        return _error(400, str(exc))
    return jsonify(result)


__all__ = ["BANK_EXTENSION", "objective_walkthrough_bp"]
