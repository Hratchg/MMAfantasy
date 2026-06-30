"""Per-request access-log middleware (API-V24-06, Phase 35).

`AccessLogMiddleware` emits exactly one JSON log record per request to the
``ufc_prediction.access`` logger with four required keys in the record's
``extra`` payload: ``path``, ``method``, ``status_code``, ``duration_ms``.

The middleware is intentionally minimal — it leans on the JSON formatter
installed by ``configure_logging()`` (Plan 35-02) to enrich records with
``request_id`` + ``partner_label`` via contextvars. That means
``RequestIDMiddleware`` MUST run BEFORE this one in the dispatch order
(i.e., be added AFTER it via ``app.add_middleware`` — Starlette stacks
LIFO) so the contextvar is bound when this middleware's ``finally`` block
fires the log record.

Behavior contract (Plan 35-06 Task 1):
- Exactly one log record per request, regardless of downstream success
  or failure (exceptions still log status_code=500 and are re-raised).
- ``duration_ms`` is a float in millisecond units (``perf_counter`` delta
  multiplied by 1000.0), rounded to 2 decimals for log noise reduction.
- Status code defaults to 500 if the wrapped handler raises (the
  exception bubbles up but the record is still emitted via ``finally``).
"""

from __future__ import annotations

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

_logger = logging.getLogger("ufc_prediction.access")


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Emit one structured access-log record per request."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        status_code = 500  # default if downstream raises
        try:
            response: Response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0
            _logger.info(
                "request",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "status_code": status_code,
                    "duration_ms": round(duration_ms, 2),
                },
            )
