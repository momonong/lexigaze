"""Fail-closed request policy for the local objective walkthrough."""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Mapping
from typing import Any


DEFAULT_WALKTHROUGH_AUTHORITY = "127.0.0.1:8098"
MAX_WALKTHROUGH_CONTENT_LENGTH = 64 * 1024
AUTHORITY_PATTERN = re.compile(r"^127\.0\.0\.1:([1-9][0-9]{0,4})$")

WALKTHROUGH_METHODS = {
    "/objective-profile": frozenset({"GET", "HEAD"}),
    "/objective-profile/assets/objective_walkthrough.css": frozenset({"GET", "HEAD"}),
    "/objective-profile/assets/objective_walkthrough.js": frozenset({"GET", "HEAD"}),
    "/api/objective-profile/health": frozenset({"GET", "HEAD"}),
    "/api/objective-profile/form": frozenset({"GET", "HEAD"}),
    "/api/objective-profile/score": frozenset({"POST"}),
}

WALKTHROUGH_SECURITY_HEADERS = {
    "Cache-Control": "no-store, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-DNS-Prefetch-Control": "off",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Content-Security-Policy": (
        "default-src 'self'; base-uri 'none'; connect-src 'self'; "
        "font-src 'self'; form-action 'none'; frame-ancestors 'none'; "
        "img-src 'self'; media-src 'none'; object-src 'none'; "
        "script-src 'self'; style-src 'self'; worker-src 'none'"
    ),
}


class WalkthroughSurfaceError(ValueError):
    """Raised when the dedicated surface configuration is unsafe."""


def normalize_walkthrough_authority(value: object) -> str:
    authority = str(value or "")
    match = AUTHORITY_PATTERN.fullmatch(authority)
    if match is None:
        raise WalkthroughSurfaceError(
            "walkthrough authority must be exact 127.0.0.1:<port>"
        )
    port = int(match.group(1))
    if not 1024 <= port <= 65535:
        raise WalkthroughSurfaceError(
            "walkthrough authority must use an unprivileged port from 1024 to 65535"
        )
    return authority


def walkthrough_origin(authority: str) -> str:
    return f"http://{normalize_walkthrough_authority(authority)}"


def is_loopback_remote(value: object) -> bool:
    try:
        return ipaddress.ip_address(str(value or "")).is_loopback
    except ValueError:
        return False


def request_policy_rejection(
    *,
    authority: str,
    remote_addr: object,
    host: object,
    path: object,
    method: object,
    query_string: bytes | bytearray | memoryview | str | None,
    origin: object = None,
    sec_fetch_site: object = None,
) -> tuple[int, str] | None:
    """Return a fail-closed rejection or ``None`` for an allowed request."""

    expected_authority = normalize_walkthrough_authority(authority)
    if not is_loopback_remote(remote_addr):
        return 403, "objective walkthrough is loopback-only"
    if str(host or "") != expected_authority:
        return 421, "objective walkthrough Host authority is invalid"
    request_path = str(path or "")
    allowed_methods = WALKTHROUGH_METHODS.get(request_path)
    if allowed_methods is None:
        return 404, "endpoint is unavailable on the objective walkthrough surface"
    request_method = str(method or "").upper()
    if request_method not in allowed_methods:
        return 405, "method is unavailable on the objective walkthrough surface"
    if query_string not in (None, b"", ""):
        return 400, "query parameters are forbidden on the objective walkthrough surface"
    supplied_origin = str(origin or "")
    if supplied_origin and supplied_origin != walkthrough_origin(expected_authority):
        return 403, "cross-origin request is forbidden"
    fetch_site = str(sec_fetch_site or "").lower()
    if fetch_site and fetch_site not in {"same-origin", "none"}:
        return 403, "cross-site request is forbidden"
    return None


def json_object_rejection(*, mimetype: object, parsed_json: Any) -> tuple[int, str] | None:
    if str(mimetype or "").lower() != "application/json":
        return 415, "Content-Type must be application/json"
    if not isinstance(parsed_json, Mapping):
        return 400, "request JSON body must be an object"
    return None


def walkthrough_security_headers() -> dict[str, str]:
    return dict(WALKTHROUGH_SECURITY_HEADERS)


__all__ = [
    "DEFAULT_WALKTHROUGH_AUTHORITY",
    "MAX_WALKTHROUGH_CONTENT_LENGTH",
    "WalkthroughSurfaceError",
    "json_object_rejection",
    "normalize_walkthrough_authority",
    "request_policy_rejection",
    "walkthrough_security_headers",
]
