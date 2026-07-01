"""FastAPI application factory.

Phase 35 Plan 35-06 — integration wave. Wires Wave 1 + Wave 2 modules
(auth, request-id, observability, rate-limit, health) into create_app(),
applies ``require_api_key`` to all /api/v1/* routers via include_router
dependencies, mounts /health + /ready at root (auth + rate-limit
exempt), env-gates /docs + /redoc + /openapi.json for prod
(API-V24-05), and binds CORS allow_origins to ``settings.cors_origins``.

Middleware order (Starlette is LIFO — last added runs OUTERMOST):
  Request → CORS → RequestID → SlowAPI → AccessLog → Route

We want CORS to be outermost so preflight OPTIONS responses bypass
everything else; RequestID must bind the contextvar before SlowAPI or
AccessLog need it; AccessLog's ``finally`` reads the contextvar after
the response (or exception) has been produced.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from ufc_prediction.api import health
from ufc_prediction.api.auth import require_api_key
from ufc_prediction.api.disclaimer import DISCLAIMER_200W
from ufc_prediction.api.middleware.access_log import AccessLogMiddleware
from ufc_prediction.api.middleware.request_id import RequestIDMiddleware
from ufc_prediction.api.rate_limit import (
    build_limiter,
    rate_limit_exceeded_handler,
    register_exempt_routes_on_limiter,
)
from ufc_prediction.api.routers import fighters, history, matchup, rankings
from ufc_prediction.api.v1 import predict as v1_predict
from ufc_prediction.api.v1.models import ProblemDetailsV13
from ufc_prediction.config import settings
from ufc_prediction.obs.logging import configure_logging
from ufc_prediction.obs.sentry import init_sentry

# Phase 52 API-V26-01 — RFC 7807 content type. Partners opt in to the
# ProblemDetails error wrapper by sending this Accept header. Anything
# else (default `*/*` or `application/json`) keeps the existing FastAPI
# `{"detail": "..."}` shape (forward-compat for v1.0.0/v1.1.0/v1.2.0).
PROBLEM_JSON_CONTENT_TYPE = "application/problem+json"


def _problem_details_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Render HTTPExceptions as RFC 7807 problem+json when partner opts in.

    Activation: ``Accept: application/problem+json`` on the inbound
    request. Otherwise FastAPI's default ``{"detail": ...}`` shape is
    preserved by returning the same JSONResponse without the
    problem+json content type.

    This handler is registered for ``HTTPException`` (both FastAPI's and
    Starlette's). Other exception types — RateLimitExceeded for example —
    keep their existing handlers (see rate_limit_exceeded_handler).
    """
    if not isinstance(exc, StarletteHTTPException):
        # Safety: never reached given the registration site below, but
        # keeps the function total in case the handler is reused.
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error"},
        )
    accept = request.headers.get("accept", "")
    status_code = exc.status_code
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    if PROBLEM_JSON_CONTENT_TYPE not in accept.lower():
        # Forward-compat: pre-v1.3.0 partners get the existing shape.
        return JSONResponse(status_code=status_code, content={"detail": exc.detail})
    problem = ProblemDetailsV13(
        type="about:blank",
        title=detail or "HTTP error",
        status=status_code,
        detail=detail,
        instance=str(request.url.path),
    )
    return JSONResponse(
        status_code=status_code,
        content=problem.model_dump(mode="json"),
        media_type=PROBLEM_JSON_CONTENT_TYPE,
    )


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    # 1. Initialize observability BEFORE app construction so any startup
    #    errors surface through configured logging + Sentry.
    configure_logging()
    init_sentry()

    # 2. Build FastAPI with env-gated docs (API-V24-05).
    #    In production, /docs, /redoc, and /openapi.json all return 404.
    docs_url = "/docs" if settings.env != "prod" else None
    redoc_url = "/redoc" if settings.env != "prod" else None
    openapi_url = "/openapi.json" if settings.env != "prod" else None
    app = FastAPI(
        title="UFC Fight Prediction API",
        version="2.3.0",
        description=DISCLAIMER_200W,
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
    )

    # 3. Rate-limit infrastructure — limiter bucketed by partner label;
    #    register exempt paths (/health + /ready) AFTER routes are mounted
    #    so Fly's probes don't burn the per-partner bucket. /docs +
    #    /redoc + /openapi.json have no route handler and are exempt
    #    naturally via slowapi's handler=None short-circuit.
    limiter = build_limiter(default_limits=["1000/hour"])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)  # type: ignore[arg-type]

    # Phase 52 API-V26-01 — RFC 7807 opt-in error wrapper for HTTPException.
    # Activates only when `Accept: application/problem+json` is on the
    # request; otherwise preserves FastAPI's default `{"detail": ...}` shape
    # (forward-compat lock for v1.0.0 / v1.1.0 / v1.2.0 partners).
    app.add_exception_handler(
        HTTPException,
        _problem_details_exception_handler,
    )
    app.add_exception_handler(
        StarletteHTTPException,
        _problem_details_exception_handler,
    )

    # 4. Middleware stack — Starlette ``add_middleware`` is LIFO, so the
    #    LAST add wraps the request OUTERMOST. The desired runtime order
    #    is: CORS → RequestID → SlowAPI → AccessLog → Route, so we add
    #    them in reverse:
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        # Default [] — operator MUST set UFC_CORS_ORIGINS in prod.
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # 5. Mount health endpoints at root (no auth, no rate limit).
    app.include_router(health.router)

    # 6. Mount /api/v1/* routers WITH require_api_key dependency.
    api_v1_deps = [Depends(require_api_key)]
    app.include_router(
        fighters.router,
        prefix="/api/v1",
        dependencies=api_v1_deps,
    )
    app.include_router(
        rankings.router,
        prefix="/api/v1",
        dependencies=api_v1_deps,
    )
    app.include_router(
        matchup.router,
        prefix="/api/v1",
        dependencies=api_v1_deps,
    )
    app.include_router(
        history.router,
        prefix="/api/v1",
        dependencies=api_v1_deps,
    )
    app.include_router(
        v1_predict.router,
        prefix="/api/v1",
        dependencies=api_v1_deps,
    )

    # 7. Register /health + /ready as rate-limit-exempt (after their
    #    routes are mounted so we can resolve their handler qualnames).
    register_exempt_routes_on_limiter(app, limiter)

    return app
