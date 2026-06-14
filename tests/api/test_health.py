"""Tests for /health + /ready endpoints (Plan 35-03, API-V24-04)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from ufc_prediction.api import health as health_module
from ufc_prediction.api.health import router as health_router


@pytest.fixture()
def app_with_health():
    app = FastAPI(version="35.0.0-test")
    app.include_router(health_router)
    return app


@pytest.fixture()
def client(app_with_health):
    with TestClient(app_with_health) as c:
        yield c


def test_health_returns_ok_and_version_without_db(client, monkeypatch):
    """liveness MUST NOT touch the DB."""
    sentinel = MagicMock()
    monkeypatch.setattr(health_module, "SessionLocal", sentinel)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "version": "35.0.0-test"}
    sentinel.assert_not_called()


def test_ready_returns_200_on_healthy_db(client, monkeypatch):
    class _FakeResult:
        def scalar(self):
            return 1

    class _FakeSession:
        def execute(self, _stmt):
            return _FakeResult()

        def close(self):
            pass

    monkeypatch.setattr(health_module, "SessionLocal", lambda: _FakeSession())
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}


def test_ready_returns_503_on_db_error(client, monkeypatch):
    def _broken_session():
        class _S:
            def execute(self, _stmt):
                raise OperationalError("stmt", {}, Exception("connection refused"))

            def close(self):
                pass

        return _S()

    monkeypatch.setattr(health_module, "SessionLocal", _broken_session)
    resp = client.get("/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "db_unavailable"
    assert "error" in body
    assert "connection refused" in body["error"] or "OperationalError" in body["error"]


def test_ready_returns_503_on_timeout(client, monkeypatch):
    """A query that hangs >1s MUST be capped by asyncio.wait_for."""

    class _SlowSession:
        def execute(self, _stmt):
            time.sleep(5)
            return MagicMock(scalar=lambda: 1)

        def close(self):
            pass

    monkeypatch.setattr(health_module, "SessionLocal", lambda: _SlowSession())

    start = time.monotonic()
    resp = client.get("/ready")
    elapsed = time.monotonic() - start

    assert resp.status_code == 503
    assert resp.json() == {"status": "db_unavailable", "error": "timeout"}
    # Must not block 5s — the 1s wait_for should fire well before.
    assert elapsed < 2.5, f"timeout was {elapsed:.2f}s (expected ~1s cap)"


def test_health_and_ready_are_auth_exempt(client, monkeypatch):
    """Both endpoints must answer without X-API-Key — no 401."""

    class _OkSession:
        def execute(self, _stmt):
            return MagicMock(scalar=lambda: 1)

        def close(self):
            pass

    monkeypatch.setattr(health_module, "SessionLocal", lambda: _OkSession())

    h = client.get("/health")  # no headers
    r = client.get("/ready")  # no headers
    assert h.status_code == 200
    assert r.status_code in (200, 503)  # 200 with our stub
