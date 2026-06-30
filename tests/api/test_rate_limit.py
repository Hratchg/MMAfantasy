"""Tests for slowapi limiter + 429 handler + exempt paths (Plan 35-03, API-V24-02)."""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from ufc_prediction.api import auth as auth_module
from ufc_prediction.api.auth import require_api_key
from ufc_prediction.api.rate_limit import (
    EXEMPT_PATHS,
    build_limiter,
    is_exempt,
    rate_limit_exceeded_handler,
)


@pytest.fixture()
def limited_app(monkeypatch):
    """FastAPI app with rate-limited /api/v1/test + exempt /health and /ready."""
    monkeypatch.setattr(
        auth_module.settings,
        "api_keys",
        ["partner-a:secret-a", "partner-b:secret-b"],
    )
    # Build a limiter with NO default_limits — SlowAPIMiddleware otherwise
    # enforces them on every route (including exempt /health and /ready).
    # Per-route decorators below enforce the limit only where intended.
    limiter = build_limiter(default_limits=[])
    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    @app.get("/api/v1/test")
    @limiter.limit("3/hour")
    async def limited_route(request: Request, _: None = Depends(require_api_key)):
        return {"ok": True}

    @app.get("/health")
    async def health(request: Request):
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request):
        return {"status": "ready"}

    return app


@pytest.fixture()
def client(limited_app):
    with TestClient(limited_app) as c:
        yield c


def test_429_after_third_request(client):
    headers = {"X-API-Key": "partner-a:secret-a"}
    for i in range(3):
        resp = client.get("/api/v1/test", headers=headers)
        assert resp.status_code == 200, f"request #{i + 1} unexpectedly 429"
    fourth = client.get("/api/v1/test", headers=headers)
    assert fourth.status_code == 429


def test_429_body_shape_and_headers(client):
    headers = {"X-API-Key": "partner-a:secret-a"}
    for _ in range(3):
        client.get("/api/v1/test", headers=headers)
    resp = client.get("/api/v1/test", headers=headers)
    assert resp.status_code == 429
    body = resp.json()
    assert body["error"] == "rate_limit_exceeded"
    assert body["limit"] == 3
    assert body["period"] == "hour"
    assert isinstance(body["retry_after_seconds"], int)
    assert body["retry_after_seconds"] > 0
    assert resp.headers.get("Retry-After") == str(body["retry_after_seconds"])


def test_different_partners_have_independent_buckets(client):
    """Partner-A burst MUST NOT consume Partner-B quota."""
    for _ in range(3):
        resp = client.get("/api/v1/test", headers={"X-API-Key": "partner-a:secret-a"})
        assert resp.status_code == 200
    # Partner-A is now exhausted; Partner-B should still have full quota.
    resp_b = client.get("/api/v1/test", headers={"X-API-Key": "partner-b:secret-b"})
    assert resp_b.status_code == 200


def test_health_and_ready_bypass_limiter(client):
    # Hit /health/ready way past the 3/hour limit — never returns 429.
    for _ in range(10):
        h = client.get("/health")
        r = client.get("/ready")
        assert h.status_code == 200
        assert r.status_code == 200


def test_exempt_paths_constant():
    expected = {"/health", "/ready", "/docs", "/redoc", "/openapi.json"}
    assert EXEMPT_PATHS == frozenset(expected)
    assert isinstance(EXEMPT_PATHS, frozenset)


def test_is_exempt_predicate():
    class _FakeUrl:
        def __init__(self, path):
            self.path = path

    class _FakeRequest:
        def __init__(self, path):
            self.url = _FakeUrl(path)

    for p in ("/health", "/ready", "/docs", "/redoc", "/openapi.json"):
        assert is_exempt(_FakeRequest(p)) is True
    assert is_exempt(_FakeRequest("/api/v1/predict")) is False
