"""Phase 52 API-V26-01 — ProblemDetails (RFC 7807) opt-in error wrapper tests.

Verifies the v1.3.0 additive contract:
  - default partners (no Accept header or `Accept: application/json`) keep
    the existing FastAPI `{"detail": "..."}` error shape
  - v1.3.0 partners opting in via `Accept: application/problem+json` get
    an RFC 7807-shaped body with `Content-Type: application/problem+json`
  - the success path is byte-untouched (covered by the existing
    `test_forward_compat_v1_2_0.py` regression; this file focuses on the
    error path only)

Forward-compat lock: v1.0.0 + v1.1.0 + v1.2.0 partners NEVER see the
ProblemDetails shape — Phase 52 is opt-in via the Accept header only.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    # Save/restore the two env vars this fixture sets so it never leaks into
    # os.environ. Leaking these previously contaminated the process-wide
    # environment (and any later test that reconstructs `Settings`), an
    # order-dependent test-isolation hazard. Record the prior state, set the
    # test values for the app under test, then restore on teardown.
    keys = ("UFC_API_KEYS", "UFC_ENV")
    values = {"UFC_API_KEYS": "test-key:test-partner", "UFC_ENV": "dev"}
    prev = {k: os.environ.get(k) for k in keys}
    for k, v in values.items():
        os.environ[k] = v
    try:
        from ufc_prediction.api.app import create_app

        with TestClient(create_app()) as test_client:
            yield test_client
    finally:
        for k in keys:
            if prev[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev[k]


def test_default_accept_preserves_existing_detail_shape(client: TestClient) -> None:
    """No Accept header → FastAPI default `{"detail": "..."}` body."""
    r = client.get(
        "/api/v1/fighters/__does_not_exist__",
        headers={"X-API-Key": "wrong-key"},
    )
    assert r.status_code == 401
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert "detail" in body
    # Ensure NOT problem+json — no top-level title/status/type fields.
    assert "title" not in body
    assert "type" not in body


def test_problem_json_accept_emits_rfc7807_body(client: TestClient) -> None:
    """Accept: application/problem+json → 7807-shaped response."""
    r = client.get(
        "/api/v1/fighters/__does_not_exist__",
        headers={
            "X-API-Key": "wrong-key",
            "Accept": "application/problem+json",
        },
    )
    assert r.status_code == 401
    assert r.headers["content-type"].startswith("application/problem+json")
    body = r.json()
    # RFC 7807 §3.1 member set
    assert body["type"] == "about:blank"
    assert body["status"] == 401
    assert "title" in body
    assert "detail" in body
    assert body["instance"] == "/api/v1/fighters/__does_not_exist__"


def test_problem_json_accept_with_extra_quality_params(client: TestClient) -> None:
    """Accept with extra params (q=0.9, etc) still routes to problem+json."""
    r = client.get(
        "/api/v1/fighters/__does_not_exist__",
        headers={
            "X-API-Key": "wrong-key",
            "Accept": "application/problem+json; q=0.9, application/json; q=0.1",
        },
    )
    assert r.status_code == 401
    assert r.headers["content-type"].startswith("application/problem+json")


def test_problem_details_model_validates() -> None:
    """ProblemDetailsV13 Pydantic model accepts RFC 7807 member set."""
    from ufc_prediction.api.v1.models import ProblemDetailsV13

    p = ProblemDetailsV13(
        title="Not Found",
        status=404,
        detail="Fighter X not in DB",
        instance="/api/v1/fighters/X",
    )
    dumped = p.model_dump(mode="json")
    assert dumped == {
        "type": "about:blank",
        "title": "Not Found",
        "status": 404,
        "detail": "Fighter X not in DB",
        "instance": "/api/v1/fighters/X",
    }


def test_problem_details_status_validation() -> None:
    """ProblemDetailsV13 enforces RFC 7807 §3.1 status range (100-599)."""
    from pydantic import ValidationError

    from ufc_prediction.api.v1.models import ProblemDetailsV13

    with pytest.raises(ValidationError):
        ProblemDetailsV13(title="x", status=99)
    with pytest.raises(ValidationError):
        ProblemDetailsV13(title="x", status=600)


def test_accept_schema_version_literal_includes_v130() -> None:
    """`accept_schema_version` Literal extended to admit '1.3.0'."""
    from ufc_prediction.api.v1.models import PredictMatchupRequestV1

    req = PredictMatchupRequestV1(
        fighter_a="A",
        fighter_b="B",
        accept_schema_version="1.3.0",
    )
    assert req.accept_schema_version == "1.3.0"
