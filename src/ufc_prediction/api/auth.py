"""API-key authentication dependency (API-V24-01, Phase 35).

`require_api_key` is a FastAPI dependency that gates protected routes. It
reads the `X-API-Key` header, performs a constant-time compare against
every entry in `settings.api_keys`, and on a match populates two
request-scoped attributes for downstream consumers:

- `request.state.partner_label` — used by slowapi key_func (rate-limit
  bucketing) and the JSON log formatter (audit trail).
- `request.state.api_key_raw` — the matched key string itself; useful
  for debugging from inside a route handler.

API-key format: each entry in `settings.api_keys` is `partner-label:rawsecret`.
First-colon split derives the partner label. Bare keys (no colon) fall back
to an auto-label `anon-<sha256[:6]>`.

401 response shape per CONTEXT decision: two distinct error codes so
partners can self-diagnose (missing vs invalid header) without leaking
which key was attempted.
"""

from __future__ import annotations

import hashlib
import secrets

from fastapi import Header, HTTPException, Request

from ufc_prediction.config import settings


def _derive_partner_label(matched_key: str) -> str:
    """Derive a stable partner label from a matched key entry.

    `partner:secret` → "partner"; bare keys get an anonymized hash label.
    """
    if ":" in matched_key:
        return matched_key.split(":", 1)[0]
    return f"anon-{hashlib.sha256(matched_key.encode()).hexdigest()[:6]}"


def require_api_key(
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> None:
    """FastAPI dependency: require a valid X-API-Key header.

    Raises 401 (missing_api_key) when the header is absent or empty.
    Raises 401 (invalid_api_key) when the value doesn't match any
    entry in `settings.api_keys`. On success, sets
    `request.state.partner_label` and `request.state.api_key_raw`.

    Uses `secrets.compare_digest` and a counter-sum loop (no
    short-circuit) to keep timing flat across attempts.
    """
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "missing_api_key",
                "detail": "X-API-Key header required",
            },
        )

    matched_key: str | None = None
    match_count = 0
    for stored in settings.api_keys:
        # secrets.compare_digest is constant-time over equal-length inputs;
        # sum-into-counter avoids short-circuit timing differences.
        if secrets.compare_digest(stored, x_api_key):
            match_count += 1
            matched_key = stored

    if match_count == 0 or matched_key is None:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "invalid_api_key",
                "detail": "X-API-Key not recognized",
            },
        )

    partner_label = _derive_partner_label(matched_key)
    request.state.partner_label = partner_label
    request.state.api_key_raw = matched_key


def partner_label_from_request(request: Request) -> str | None:
    """Return the partner label set by `require_api_key`, or None.

    Used by slowapi key_func (Plan 35-03) to bucket rate limits per
    partner. Returns None when no auth dependency has run on this
    request (slowapi falls back to "anonymous").
    """
    return getattr(request.state, "partner_label", None)
