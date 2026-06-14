"""Request-ID middleware + contextvars for structured logging (API-V24-06).

`RequestIDMiddleware` ensures every request gets a UUIDv4 correlation id
(or echoes a client-supplied `X-Request-ID` header verbatim for partner
correlation flows). The id is bound to a contextvar `request_id_ctxvar`
that the JSON log formatter reads, so every log record emitted during
the request scope is automatically enriched.

A second contextvar `partner_label_ctxvar` is exposed here (set by the
auth dependency, read by the formatter) so the partner label flows
through logs without coupling to per-request state.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

request_id_ctxvar: ContextVar[str] = ContextVar("request_id", default="-")
partner_label_ctxvar: ContextVar[str | None] = ContextVar(
    "partner_label", default=None
)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Generate/echo X-Request-ID and bind it to request_id_ctxvar.

    Behavior:
    - If the client supplies `X-Request-ID`, the server echoes it
      verbatim (partner correlation flows through).
    - Otherwise a fresh uuid4 is generated.
    - The id is bound to `request_id_ctxvar` for the duration of the
      request via a token; reset in `finally` so async/threadpool
      reuse doesn't leak ids between requests.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        client_id = request.headers.get("X-Request-ID", "").strip()
        request_id = client_id or str(uuid.uuid4())
        token = request_id_ctxvar.set(request_id)
        request.state.request_id = request_id
        try:
            response: Response = await call_next(request)
        finally:
            request_id_ctxvar.reset(token)
        response.headers["X-Request-ID"] = request_id
        return response
