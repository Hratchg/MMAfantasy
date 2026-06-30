"""slowapi rate-limit configuration (API-V24-02, Phase 35).

Build a partner-bucketed Limiter (key_func reads `request.state.partner_label`
set by `require_api_key`) backed by in-memory storage. Plan 35-06 wires the
limiter onto the FastAPI app and registers the 429 handler — this module
only constructs the primitives.

Locked decisions (CONTEXT.md "Rate-Limit Backend"):
- Storage: in-memory ("memory://"), matches Fly single-instance deploy
- Default limit: "1000/hour" per partner key
- Exempt paths: /health, /ready, /docs, /redoc, /openapi.json
- 429 body: {"error":"rate_limit_exceeded","limit":N,"period":"hour","retry_after_seconds":N}
"""

from __future__ import annotations

import logging
import re

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

from ufc_prediction.api.auth import partner_label_from_request

logger = logging.getLogger(__name__)

EXEMPT_PATHS: frozenset[str] = frozenset({"/health", "/ready", "/docs", "/redoc", "/openapi.json"})

# Period -> seconds, used to derive Retry-After when slowapi's exception text
# is well-formed. Falls back to 3600 (1 hour) for unknown periods.
_PERIOD_SECONDS: dict[str, int] = {
    "second": 1,
    "minute": 60,
    "hour": 3600,
    "day": 86400,
}


def is_exempt(request: Request) -> bool:
    """Return True for paths that bypass rate limiting (health probes, docs)."""
    return request.url.path in EXEMPT_PATHS


def _key_func(request: Request) -> str:
    """slowapi key_func — bucket by partner label, not IP.

    Falls back to "anonymous" when the auth dependency hasn't populated
    `request.state.partner_label` (defensive — normally a 401 from
    `require_api_key` would short-circuit before slowapi runs).
    """
    return partner_label_from_request(request) or "anonymous"


def build_limiter(default_limits: list[str] | None = None) -> Limiter:
    """Construct a Limiter bucketing by partner label.

    Default limit per CONTEXT decision is "1000/hour"; tests pass override
    values (e.g. "3/hour") for fast verification.

    Path-level exemption (CONTEXT — Fly probes against /health + /ready
    must not burn the per-partner bucket) is handled by
    ``register_exempt_routes_on_limiter()`` after the routes are mounted;
    slowapi tracks exempted handlers by their qualified name on the
    ``Limiter._exempt_routes`` set.
    """
    if default_limits is None:
        default_limits = ["1000/hour"]
    return Limiter(
        key_func=_key_func,
        default_limits=list(default_limits),
        storage_uri="memory://",
    )


def register_exempt_routes_on_limiter(app: FastAPI, limiter: Limiter) -> None:
    """Mark routes whose path is in ``EXEMPT_PATHS`` as rate-limit exempt.

    slowapi tracks exemption by qualified handler name in
    ``limiter._exempt_routes``. We resolve handler names for the mounted
    routes whose path lives in ``EXEMPT_PATHS`` and add them to that set
    — equivalent to applying the ``@limiter.exempt`` decorator at route
    definition time, but performed centrally in ``create_app()`` so the
    health module need not import the limiter.

    /docs, /redoc, /openapi.json are NOT FastAPI routes (they're handled
    via the FastAPI factory's docs_url/redoc_url/openapi_url) so they
    aren't iterated here — but the SlowAPIMiddleware ``_find_route_handler``
    returns ``None`` for handler-less paths and ``_should_exempt`` already
    treats handler=None as exempt, so they bypass naturally.
    """
    for route in app.routes:
        # Starlette ``Route`` exposes ``path`` and ``endpoint`` (the
        # function/coroutine); other route types (Mount, WebSocket) lack
        # ``endpoint``, so guard accordingly.
        path = getattr(route, "path", None)
        endpoint = getattr(route, "endpoint", None)
        if path in EXEMPT_PATHS and endpoint is not None:
            qualname = f"{endpoint.__module__}.{endpoint.__qualname__}"
            limiter._exempt_routes.add(qualname)


def _parse_limit_detail(detail: str) -> tuple[int, str]:
    """Parse slowapi exception detail like '3 per 1 hour' → (3, 'hour').

    Slowapi formats exception details consistently; tolerate variations
    by falling back to (1000, 'hour') if parsing fails.
    """
    match = re.match(r"^\s*(\d+)\s+per\s+\d+\s+(\w+)\s*$", detail)
    if match:
        return int(match.group(1)), match.group(2).rstrip("s")
    # Alternate shape: "3 per hour"
    match = re.match(r"^\s*(\d+)\s+per\s+(\w+)\s*$", detail)
    if match:
        return int(match.group(1)), match.group(2).rstrip("s")
    return 1000, "hour"


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Convert slowapi's RateLimitExceeded into the locked 429 JSON body shape.

    SYNC by design — ``SlowAPIMiddleware.sync_check_limits`` falls back to
    slowapi's default text handler when this is a coroutine (see slowapi
    middleware.py:73-75). Keeping it sync ensures the middleware-path 429s
    (raised by ``default_limits``) also get the structured JSON body.
    """
    try:
        limit_int, period = _parse_limit_detail(str(exc.detail))
    except Exception:
        logger.warning("rate_limit_exceeded_handler: detail parse failed", exc_info=True)
        limit_int, period = 1000, "hour"
    retry_after = _PERIOD_SECONDS.get(period, 3600)
    body = {
        "error": "rate_limit_exceeded",
        "limit": limit_int,
        "period": period,
        "retry_after_seconds": retry_after,
    }
    return JSONResponse(
        status_code=429,
        content=body,
        headers={"Retry-After": str(retry_after)},
    )
