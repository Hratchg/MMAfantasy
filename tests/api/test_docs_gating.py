"""Env-gated /docs + /redoc + /openapi.json (Plan 35-06 Task 4, API-V24-05).

When ``settings.env == "prod"``, FastAPI is constructed with
``docs_url=None``, ``redoc_url=None``, and ``openapi_url=None`` — so all
three routes return 404. In ``dev`` and ``staging``, the routes are
mounted and return 200.

7 behaviors covered:
  1. prod /docs → 404
  2. prod /redoc → 404
  3. prod /openapi.json → 404
  4. dev /docs → 200 (Swagger HTML)
  5. dev /redoc → 200 (Redoc HTML)
  6. dev /openapi.json → 200 (OpenAPI JSON)
  7. staging /docs → 200 (staging is dev-equivalent for docs gate)
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ufc_prediction.api import app as app_module
from ufc_prediction.api.app import create_app


def _build_client(monkeypatch, env: str) -> TestClient:
    """Monkey-patch settings.env across all import sites + rebuild app."""
    # Patch on every module that imported settings at module load time.
    monkeypatch.setattr(app_module.settings, "env", env)
    from ufc_prediction import config as config_module
    monkeypatch.setattr(config_module.settings, "env", env)
    # Also patch CORS to avoid empty-origins surprises in OPTIONS calls,
    # and keep sentry_dsn=None so no init noise.
    monkeypatch.setattr(app_module.settings, "cors_origins", [])
    monkeypatch.setattr(config_module.settings, "cors_origins", [])
    monkeypatch.setattr(app_module.settings, "sentry_dsn", None)
    monkeypatch.setattr(config_module.settings, "sentry_dsn", None)

    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


# ─── prod: all three routes 404 ────────────────────────────────────────────


def test_prod_docs_returns_404(monkeypatch):
    """Test 1: settings.env="prod" → GET /docs returns 404."""
    c = _build_client(monkeypatch, "prod")
    resp = c.get("/docs")
    assert resp.status_code == 404


def test_prod_redoc_returns_404(monkeypatch):
    """Test 2: settings.env="prod" → GET /redoc returns 404."""
    c = _build_client(monkeypatch, "prod")
    resp = c.get("/redoc")
    assert resp.status_code == 404


def test_prod_openapi_json_returns_404(monkeypatch):
    """Test 3: settings.env="prod" → GET /openapi.json returns 404."""
    c = _build_client(monkeypatch, "prod")
    resp = c.get("/openapi.json")
    assert resp.status_code == 404


# ─── dev: all three routes 200 ─────────────────────────────────────────────


def test_dev_docs_returns_200_html(monkeypatch):
    """Test 4: settings.env="dev" → GET /docs returns 200 + Swagger HTML."""
    c = _build_client(monkeypatch, "dev")
    resp = c.get("/docs")
    assert resp.status_code == 200
    # Swagger UI page; content-type starts with text/html.
    assert resp.headers["content-type"].startswith("text/html"), (
        resp.headers["content-type"]
    )


def test_dev_redoc_returns_200_html(monkeypatch):
    """Test 5: settings.env="dev" → GET /redoc returns 200 + Redoc HTML."""
    c = _build_client(monkeypatch, "dev")
    resp = c.get("/redoc")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html"), (
        resp.headers["content-type"]
    )


def test_dev_openapi_json_returns_200_valid_json(monkeypatch):
    """Test 6: settings.env="dev" → GET /openapi.json returns 200 + valid JSON."""
    c = _build_client(monkeypatch, "dev")
    resp = c.get("/openapi.json")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json"), (
        resp.headers["content-type"]
    )
    body = resp.json()
    # Sanity — must look like an OpenAPI 3.1.0 spec.
    assert body["openapi"].startswith("3.")
    assert "paths" in body


# ─── staging: equivalent to dev for docs gating ────────────────────────────


def test_staging_docs_returns_200(monkeypatch):
    """Test 7: settings.env="staging" → /docs returns 200 (dev-equivalent)."""
    c = _build_client(monkeypatch, "staging")
    resp = c.get("/docs")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html"), (
        resp.headers["content-type"]
    )
