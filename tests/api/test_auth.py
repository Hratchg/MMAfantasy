"""Tests for require_api_key dependency (Plan 35-02, API-V24-01)."""

from __future__ import annotations

import inspect

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from ufc_prediction.api import auth as auth_module
from ufc_prediction.api.auth import (
    partner_label_from_request,
    require_api_key,
)


@pytest.fixture()
def auth_app(monkeypatch):
    """Minimal FastAPI app with one auth-gated route.

    Configures settings.api_keys with two keys (partner-prefixed + bare).
    """
    monkeypatch.setattr(
        auth_module.settings,
        "api_keys",
        ["acme:secret-abc", "legacy-raw-key"],
    )
    app = FastAPI()

    @app.get("/test")
    async def test_route(request: Request, _: None = Depends(require_api_key)):
        return {
            "partner_label": getattr(request.state, "partner_label", None),
            "api_key_raw": getattr(request.state, "api_key_raw", None),
            "label_helper": partner_label_from_request(request),
        }

    return app


@pytest.fixture()
def auth_client(auth_app):
    with TestClient(auth_app) as c:
        yield c


def test_missing_api_key_returns_401_with_structured_body(auth_client):
    resp = auth_client.get("/test")
    assert resp.status_code == 401
    body = resp.json()
    assert body["detail"] == {
        "error": "missing_api_key",
        "detail": "X-API-Key header required",
    }


def test_invalid_api_key_returns_401_with_structured_body(auth_client):
    resp = auth_client.get("/test", headers={"X-API-Key": "no-such-key"})
    assert resp.status_code == 401
    body = resp.json()
    assert body["detail"] == {
        "error": "invalid_api_key",
        "detail": "X-API-Key not recognized",
    }


def test_valid_partner_prefixed_key_sets_partner_label(auth_client):
    resp = auth_client.get("/test", headers={"X-API-Key": "acme:secret-abc"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["partner_label"] == "acme"
    assert body["api_key_raw"] == "acme:secret-abc"
    assert body["label_helper"] == "acme"


def test_valid_bare_key_gets_anon_label(auth_client):
    resp = auth_client.get("/test", headers={"X-API-Key": "legacy-raw-key"})
    assert resp.status_code == 200
    body = resp.json()
    label = body["partner_label"]
    assert label is not None
    assert label.startswith("anon-")
    # SHA-prefix is 6 hex chars after "anon-".
    assert len(label) == len("anon-") + 6
    assert all(c in "0123456789abcdef" for c in label[len("anon-"):])


def test_compare_digest_present_in_source():
    src = inspect.getsource(require_api_key)
    assert "compare_digest" in src, (
        "require_api_key must use secrets.compare_digest for constant-time compare"
    )


def test_partner_label_helper_returns_none_when_unset():
    """Outside a Depends(require_api_key) chain, helper returns None."""

    class _FakeState:
        pass

    class _FakeRequest:
        state = _FakeState()

    assert partner_label_from_request(_FakeRequest()) is None
