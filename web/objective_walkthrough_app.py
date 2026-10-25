"""Dedicated Flask app for the local, no-store objective walkthrough."""

from __future__ import annotations

from collections.abc import Mapping

from flask import Flask, jsonify, request

from core.objective_profile_walkthrough import load_walkthrough_bank
from web.objective_walkthrough_surface import (
    DEFAULT_WALKTHROUGH_AUTHORITY,
    MAX_WALKTHROUGH_CONTENT_LENGTH,
    normalize_walkthrough_authority,
    request_policy_rejection,
    walkthrough_security_headers,
)
from web.routes.objective_walkthrough import (
    BANK_EXTENSION,
    objective_walkthrough_bp,
)


def create_objective_walkthrough_app(
    config: Mapping[str, object] | None = None,
) -> Flask:
    """Create a route-minimal app that has no persistence endpoints."""

    app = Flask(__name__, static_folder=None, template_folder="templates")
    app.config.update(
        {
            "MAX_CONTENT_LENGTH": MAX_WALKTHROUGH_CONTENT_LENGTH,
            "LEXIGAZE_WALKTHROUGH_AUTHORITY": DEFAULT_WALKTHROUGH_AUTHORITY,
        }
    )
    if config:
        app.config.update(config)
    authority = normalize_walkthrough_authority(
        app.config["LEXIGAZE_WALKTHROUGH_AUTHORITY"]
    )
    app.config["LEXIGAZE_WALKTHROUGH_AUTHORITY"] = authority
    app.extensions[BANK_EXTENSION] = load_walkthrough_bank()

    @app.before_request
    def restrict_surface():
        rejection = request_policy_rejection(
            authority=authority,
            remote_addr=request.remote_addr,
            host=request.host,
            path=request.path,
            method=request.method,
            query_string=request.query_string,
            origin=request.headers.get("Origin"),
            sec_fetch_site=request.headers.get("Sec-Fetch-Site"),
        )
        if rejection is None:
            return None
        status, detail = rejection
        return jsonify(
            {
                "ok": False,
                "classification": "objective_walkthrough_surface_rejected",
                "error": detail,
                "research_claim_authorized": False,
                "persisted": False,
            }
        ), status

    @app.after_request
    def secure_surface(response):
        for name, value in walkthrough_security_headers().items():
            response.headers[name] = value
        return response

    @app.errorhandler(413)
    def reject_oversized_request(_error):
        return jsonify(
            {
                "ok": False,
                "classification": "objective_walkthrough_payload_too_large",
                "error": "request exceeds the 64 KiB walkthrough limit",
                "research_claim_authorized": False,
                "persisted": False,
            }
        ), 413

    app.register_blueprint(objective_walkthrough_bp)
    return app


__all__ = ["create_objective_walkthrough_app"]
