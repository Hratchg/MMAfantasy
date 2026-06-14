"""End-to-end integration smoke for the fully-wired create_app() factory
(Plan 35-06 Task 3).

Covers the 8 behaviors enumerated in 35-06-PLAN.md `<behavior>`:
  1. /api/v1/* without X-API-Key → 401
  2. /health is auth-exempt
  3. /ready is auth-exempt (status_code may be 200 or 503 depending on DB)
  4. /api/v1/* with valid X-API-Key → 200 + X-Request-ID
  5. Rate-limit 4th request → 429 with structured body + Retry-After
  6. /health exempt from rate limit (1000+ pings, no 429)
  7. CORS allow_origins drives header presence
  8. Access log records request_id/path/method/status_code/duration_ms

The fixtures monkey-patch settings AND rebuild the FastAPI app, then
override the rate-limit Limiter onto app.state.limiter so the test can
trigger 429 within a few requests instead of needing to issue 1001
calls.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from ufc_prediction.api import app as app_module
from ufc_prediction.api import auth as auth_module
from ufc_prediction.api import rate_limit as rate_limit_module
from ufc_prediction.api.app import create_app
from ufc_prediction.api.deps import get_db
from ufc_prediction.api.rate_limit import (
    build_limiter,
    register_exempt_routes_on_limiter,
)


_FAKE_PREDICT_RESULT = {
    "fighter_a": "Khabib Nurmagomedov",
    "fighter_b": "Conor McGregor",
    "win_probability": 0.6234,
    "base_prob": 0.6234,
    "meta_prob": None,
    "meta_learner_version": None,
    "meta_skipped_reason": "no_meta_artifact",
    "prediction_metadata": {
        "odds_source": "live",
        "odds_timestamp_iso": "2018-10-06T22:00:00+00:00",
        "fighter_a_n_ufc_fights": 10,
        "fighter_b_n_ufc_fights": 21,
        "is_debutant_either": False,
        "calibration_slice_brier": 0.19,
    },
}


class _FakePredictor:
    """In-memory fake mirroring ModelPredictor.predict()."""

    def predict(
        self, db, fighter_a_name, fighter_b_name, *, event_date=None, refresh=False,
    ):
        return {
            **_FAKE_PREDICT_RESULT,
            "fighter_a": fighter_a_name,
            "fighter_b": fighter_b_name,
        }


# ─── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def patched_settings(monkeypatch):
    """Override settings for integration tests — keys, CORS, env, sentry."""
    test_keys = ["test-partner:secret-abc"]
    test_cors = ["https://partner-a.example.com"]

    # Patch settings on every module that imported it at module load time.
    for mod in (
        auth_module,
        rate_limit_module,
        app_module,
    ):
        if hasattr(mod, "settings"):
            monkeypatch.setattr(mod.settings, "api_keys", test_keys)
            monkeypatch.setattr(mod.settings, "cors_origins", test_cors)
            monkeypatch.setattr(mod.settings, "env", "dev")
            monkeypatch.setattr(mod.settings, "sentry_dsn", None)
    # Also patch the canonical module so freshly-importing code sees them.
    from ufc_prediction import config as config_module

    monkeypatch.setattr(config_module.settings, "api_keys", test_keys)
    monkeypatch.setattr(config_module.settings, "cors_origins", test_cors)
    monkeypatch.setattr(config_module.settings, "env", "dev")
    monkeypatch.setattr(config_module.settings, "sentry_dsn", None)

    return {
        "api_keys": test_keys,
        "cors_origins": test_cors,
    }


@pytest.fixture()
def app(patched_settings):
    """Fresh app per-test (clean state — limiter buckets, contextvars)."""
    app = create_app()

    # Override DB session so /api/v1/* routes that need a session see a
    # benign stub instead of a real Postgres connection. The integration
    # smoke doesn't exercise DB code paths.
    def _fake_db():
        yield None

    app.dependency_overrides[get_db] = _fake_db
    return app


@pytest.fixture()
def client(app):
    # raise_server_exceptions=False so 500s from routes that hit the
    # stubbed DB session surface as a 500 response (good for assertions)
    # instead of propagating up and failing the test outright. The
    # integration smoke is about the middleware/auth/limiter stack, not
    # the DB-bound route bodies.
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ─── Tests ────────────────────────────────────────────────────────────────


def test_api_v1_returns_401_without_api_key(client):
    """Test 1: missing X-API-Key → 401 with structured error body."""
    resp = client.get("/api/v1/fighters/Jones")
    assert resp.status_code == 401, resp.text
    body = resp.json()
    # FastAPI HTTPException wraps the detail dict in {"detail": {...}}.
    detail = body.get("detail")
    assert isinstance(detail, dict), body
    assert detail["error"] == "missing_api_key"


def test_health_is_auth_exempt(client):
    """Test 2: /health requires no X-API-Key."""
    resp = client.get("/health")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok"


def test_ready_is_auth_exempt(client):
    """Test 3: /ready requires no X-API-Key (200 or 503 OK)."""
    resp = client.get("/ready")
    # /ready may return 503 if no DB is wired (test fixtures don't seed
    # SessionLocal); both are auth-exempt-correct.
    assert resp.status_code in (200, 503), resp.text


def _predict_post_body() -> dict:
    return {
        "fighter_a": "Khabib Nurmagomedov",
        "fighter_b": "Conor McGregor",
        "event_date": "2018-10-06",
        "accept_schema_version": "1.2.0",
    }


def test_api_v1_succeeds_with_valid_key_and_sets_request_id_header(
    client, patched_settings,
):
    """Test 4: valid X-API-Key → 200 + X-Request-ID response header."""
    headers = {"X-API-Key": patched_settings["api_keys"][0]}
    with patch(
        "ufc_prediction.api.v1.predict._get_predictor",
        return_value=_FakePredictor(),
    ):
        resp = client.post(
            "/api/v1/predict", json=_predict_post_body(), headers=headers,
        )
    assert resp.status_code == 200, resp.text
    rid = resp.headers.get("X-Request-ID")
    assert rid, "X-Request-ID header missing"
    assert "-" in rid and len(rid) >= 32, rid


def test_rate_limit_429_after_quota_exceeded(app, patched_settings):
    """Test 5: lower limit to 3/hour, 4th call → 429 with structured body."""
    test_limiter = build_limiter(default_limits=["3/hour"])
    app.state.limiter = test_limiter
    register_exempt_routes_on_limiter(app, test_limiter)

    headers = {"X-API-Key": patched_settings["api_keys"][0]}
    with TestClient(app, raise_server_exceptions=False) as c, patch(
        "ufc_prediction.api.v1.predict._get_predictor",
        return_value=_FakePredictor(),
    ):
        for i in range(3):
            r = c.post(
                "/api/v1/predict", json=_predict_post_body(), headers=headers,
            )
            assert r.status_code != 429, (
                f"request #{i+1} unexpectedly 429: {r.text}"
            )
        fourth = c.post(
            "/api/v1/predict", json=_predict_post_body(), headers=headers,
        )
        assert fourth.status_code == 429, fourth.text
        body = fourth.json()
        assert body["error"] == "rate_limit_exceeded"
        assert body["limit"] == 3
        assert body["period"] == "hour"
        assert isinstance(body["retry_after_seconds"], int)
        assert fourth.headers.get("Retry-After") == str(
            body["retry_after_seconds"],
        )


def test_health_bypasses_rate_limit(app, patched_settings):
    """Test 6: /health 1500 times never 429 (path is rate-limit-exempt)."""
    # Use a tiny limit so the test would FAIL if exemption weren't wired.
    test_limiter = build_limiter(default_limits=["3/hour"])
    app.state.limiter = test_limiter
    register_exempt_routes_on_limiter(app, test_limiter)

    with TestClient(app) as c:
        # We use 50 calls instead of 1500 — the assertion just needs to
        # exceed the 3/hour bucket; 50 >> 3.
        for i in range(50):
            r = c.get("/health")
            assert r.status_code == 200, (
                f"/health request #{i+1} unexpectedly {r.status_code}"
            )


def test_cors_preflight_allows_configured_origin(client):
    """Test 7: OPTIONS preflight from allowed origin → allow-origin header.
    From an un-allowed origin, the header is omitted (or echoes differently)."""
    headers = {
        "Origin": "https://partner-a.example.com",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "X-API-Key,Content-Type",
    }
    resp = client.options("/api/v1/predict", headers=headers)
    # Allowed origin: CORS middleware sets access-control-allow-origin.
    assert resp.headers.get("access-control-allow-origin") == (
        "https://partner-a.example.com"
    ), resp.headers

    bad_headers = {
        "Origin": "https://evil.example.com",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "X-API-Key,Content-Type",
    }
    resp2 = client.options("/api/v1/predict", headers=bad_headers)
    # Un-allowed origin: CORS middleware omits allow-origin header
    # (or returns a non-matching value); presence-of-header check is
    # sufficient.
    aco = resp2.headers.get("access-control-allow-origin")
    assert aco != "https://evil.example.com", (
        f"un-allowed origin reflected in allow-origin header: {aco!r}"
    )


def test_access_log_emits_structured_record(client, patched_settings, caplog):
    """Test 8: log capture during a successful request emits at least one
    record on the ufc_prediction.access logger with path/method/
    status_code/duration_ms in extra."""
    caplog.set_level(logging.INFO, logger="ufc_prediction.access")
    headers = {"X-API-Key": patched_settings["api_keys"][0]}
    with patch(
        "ufc_prediction.api.v1.predict._get_predictor",
        return_value=_FakePredictor(),
    ):
        resp = client.post(
            "/api/v1/predict", json=_predict_post_body(), headers=headers,
        )
    assert resp.status_code == 200, resp.text

    access_records = [
        r for r in caplog.records if r.name == "ufc_prediction.access"
    ]
    assert len(access_records) >= 1, (
        f"expected >=1 access record, got {len(access_records)}"
    )
    rec = access_records[-1]
    assert rec.path == "/api/v1/predict"
    assert rec.method == "POST"
    assert rec.status_code == 200
    assert hasattr(rec, "duration_ms")
    assert isinstance(rec.duration_ms, float)
