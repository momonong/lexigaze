"""Dedicated loopback-only app for the ephemeral learner workspace."""

from __future__ import annotations

import ipaddress
import re
import time
from collections.abc import Mapping
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from core.learner_workspace import DEFAULT_MAX_SESSIONS, LearnerWorkspaceError, LearnerWorkspaceStore
from web.objective_walkthrough_surface import walkthrough_security_headers

STORE_EXTENSION = "lexigaze_learner_workspace_store"
API = "/api/learner-workspace"
ASSETS = {"learner_workspace.js": "text/javascript", "learner_workspace.css": "text/css"}
ALLOWED_METHODS = {
    "/learner-workspace": {"GET", "HEAD"},
    **{f"/learner-workspace/assets/{name}": {"GET", "HEAD"} for name in ASSETS},
    f"{API}/health": {"GET", "HEAD"},
    f"{API}/session": {"POST", "DELETE"},
    f"{API}/state": {"GET", "HEAD"},
    f"{API}/advance": {"POST"},
}


def create_learner_workspace_app(config: Mapping[str, object] | None = None) -> Flask:
    app = Flask(__name__, static_folder=None, template_folder="templates")
    app.config.update(
        MAX_CONTENT_LENGTH=64 * 1024,
        LEXIGAZE_LEARNER_AUTHORITY="127.0.0.1:8097",
        LEXIGAZE_LEARNER_CLOCK=time.time,
        LEXIGAZE_LEARNER_MAX_SESSIONS=DEFAULT_MAX_SESSIONS,
    )
    if config:
        app.config.update(config)
    authority = str(app.config["LEXIGAZE_LEARNER_AUTHORITY"])
    match = re.fullmatch(r"127\.0\.0\.1:([1-9][0-9]{0,4})", authority)
    if match is None or not 1024 <= int(match.group(1)) <= 65535:
        raise ValueError("learner authority must be exact 127.0.0.1:<unprivileged port>")
    store = LearnerWorkspaceStore(
        clock=app.config["LEXIGAZE_LEARNER_CLOCK"],
        max_sessions=app.config["LEXIGAZE_LEARNER_MAX_SESSIONS"],
    )
    app.extensions[STORE_EXTENSION] = store

    def error(detail: str, status: int):
        return jsonify(ok=False, error=detail, persisted=False, research_evidence=False), status

    @app.before_request
    def restrict_surface():
        try:
            loopback = ipaddress.ip_address(request.remote_addr or "").is_loopback
        except ValueError:
            loopback = False
        if not loopback:
            return error("learner workspace is loopback-only", 403)
        if request.host != authority:
            return error("learner workspace Host authority is invalid", 421)
        methods = ALLOWED_METHODS.get(request.path)
        if methods is None:
            return error("endpoint is unavailable", 404)
        if request.method not in methods:
            return error("method is unavailable", 405)
        if request.query_string:
            return error("query parameters are forbidden", 400)
        origin = request.headers.get("Origin", "")
        if origin and origin != f"http://{authority}":
            return error("cross-origin request is forbidden", 403)
        fetch_site = request.headers.get("Sec-Fetch-Site", "").lower()
        if fetch_site and fetch_site not in {"same-origin", "none"}:
            return error("cross-site request is forbidden", 403)
        return None

    @app.after_request
    def secure_surface(response):
        for name, value in walkthrough_security_headers().items():
            response.headers[name] = value
        return response

    @app.errorhandler(LearnerWorkspaceError)
    def reject_contract(exc):
        return error(str(exc), exc.status)

    @app.errorhandler(413)
    def reject_large_request(_exc):
        return error("request exceeds the 64 KiB preview limit", 413)

    def body():
        if request.mimetype != "application/json":
            raise LearnerWorkspaceError("Content-Type must be application/json", 415)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise LearnerWorkspaceError("request JSON must be an object")
        return payload

    def token():
        return request.headers.get("X-Learner-Session")

    @app.get("/learner-workspace")
    def page():
        return render_template("learner_workspace.html")

    @app.get("/learner-workspace/assets/<filename>")
    def asset(filename):
        if filename not in ASSETS:
            return error("asset is unavailable", 404)
        path = Path(__file__).resolve().parent / "static" / filename
        return app.response_class(path.read_bytes(), mimetype=ASSETS[filename])

    @app.get(f"{API}/health")
    def health():
        return jsonify(
            ok=True, service="learner-workspace-owner-preview-v1",
            scope="owner_developer_preview", storage="process_memory_only",
            camera="disabled", gpu="not_applicable", model_inference="none",
            research_evidence=False, session_ttl_seconds=1800,
        )

    @app.post(f"{API}/session")
    def create_session():
        if body() != {}:
            raise LearnerWorkspaceError("session creation accepts an empty object only")
        return jsonify(ok=True, **store.create())

    @app.delete(f"{API}/session")
    def delete_session():
        store.delete(token())
        return jsonify(ok=True)

    @app.get(f"{API}/state")
    def state():
        return jsonify(ok=True, state=store.get_state(token()))

    @app.post(f"{API}/advance")
    def advance():
        return jsonify(ok=True, state=store.advance(token(), body()))

    return app


__all__ = ["create_learner_workspace_app"]
