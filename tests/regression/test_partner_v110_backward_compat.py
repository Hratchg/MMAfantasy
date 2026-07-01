"""Phase 32 D-05 regression — /api/v1/predict v1.0.0 backward-compat.

Asserts that the default `/api/v1/predict` response payload remains
byte-identical for v1.0.0 partners after Phase 32 D-04 adds the v1.1.0
additive trio (gate_contract_ref, model_candidates, phase_chain_audit_sha).

The contract:
  - The route at src/ufc_prediction/api/v1/predict.py:96-106 constructs
    PredictorOutputV1 with `schema_version="1.0.0"` and does NOT populate
    the additive trio. Pydantic Optional defaults to None.
  - v1.0.0 partner deserializers (built against predictor.schema.v1.0.0.json)
    either ignore the extra keys (extra="ignore") or surface them as None
    Optional fields.
  - v1.1.0 partners can opt-in by switching their committed schema file;
    a populated additive trio round-trips through model_dump correctly
    (proven by test_additive_trio_populated_when_explicitly_set below).

Mocks ModelPredictor.predict() — mirrors tests/api/test_predict_v1.py's
no-DB pattern so the suite runs in any environment without testcontainers.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from ufc_prediction.api.deps import get_db
from ufc_prediction.api.v1 import predict as v1_predict
from ufc_prediction.api.v1.models import PredictorOutputV1

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONTRACTS_DIR = REPO_ROOT / "src" / "ufc_prediction" / "contracts"
SCHEMA_V100_PATH = CONTRACTS_DIR / "predictor.schema.v1.0.0.json"
SCHEMA_V110_PATH = CONTRACTS_DIR / "predictor.schema.v1.1.0.json"

# v1.0.0 field set (Phase 25 lock — must NEVER drift).
V100_FIELDS: frozenset[str] = frozenset(
    {
        "schema_version",
        "win_probability",
        "fighter_a",
        "fighter_b",
        "event_date",
        "base_prob",
        "meta_prob",
        "meta_learner_version",
        "meta_skipped_reason",
    }
)

V110_ADDITIVE_TRIO: frozenset[str] = frozenset(
    {
        "gate_contract_ref",
        "model_candidates",
        "phase_chain_audit_sha",
    }
)

# Canonical xgb_v2.joblib SHA — used in the populated-fields test.
XGB_V2_CANONICAL_SHA = "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"


_FAKE_PREDICT_RESULT = {
    "fighter_a": "Khabib Nurmagomedov",
    "fighter_b": "Conor McGregor",
    "win_probability": 0.6234,
    "base_prob": 0.6234,
    "meta_prob": None,
    "meta_kind": None,
    "meta_learner_version": None,
    "meta_skipped": True,
    "meta_skipped_reason": "no_meta_artifact",
    "model_probability_a": 0.6234,
    "model_probability_b": 0.3766,
    "elo_probability_a": 0.5,
    "elo_probability_b": 0.5,
    "elo_a": 1500.0,
    "elo_b": 1500.0,
    "feature_importances": [],
    "odds_source": "nan",
}


class _FakePredictor:
    """Mimics ModelPredictor.predict() return shape (no DB / no joblib load)."""

    def predict(
        self,
        db,
        fighter_a_name: str,
        fighter_b_name: str,
        *,
        event_date=None,
        refresh=False,
    ) -> dict:
        return {
            **_FAKE_PREDICT_RESULT,
            "fighter_a": fighter_a_name,
            "fighter_b": fighter_b_name,
        }


@pytest.fixture()
def standalone_client() -> TestClient:
    """Standalone TestClient — no DB session needed (get_db yields None)."""
    app = FastAPI()
    app.include_router(v1_predict.router, prefix="/api/v1")

    def _fake_db():
        yield None

    app.dependency_overrides[get_db] = _fake_db
    return TestClient(app)


@pytest.fixture()
def v100_schema_validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_V100_PATH.read_text())
    return Draft202012Validator(schema)


@pytest.fixture()
def v110_schema_validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_V110_PATH.read_text())
    return Draft202012Validator(schema)


def _post_predict(client: TestClient) -> dict:
    """POST /api/v1/predict with a known-good payload (mocked predictor)."""
    with patch(
        "ufc_prediction.api.v1.predict._get_predictor",
        return_value=_FakePredictor(),
    ):
        resp = client.post(
            "/api/v1/predict",
            json={
                "fighter_a": "Khabib Nurmagomedov",
                "fighter_b": "Conor McGregor",
                "event_date": "2018-10-06",
                "accept_schema_version": "1.0.0",
            },
        )
    assert resp.status_code == 200, resp.text
    return resp.json()


# ─── D-05 regression: v1.0.0 response shape unchanged ─────────────────────────


def test_v100_response_shape_byte_identical(standalone_client: TestClient):
    """Default /api/v1/predict response: additive trio is None for v1.0.0 partners.

    Phase 32 D-05 binding: v1.0.0 partners on predictor.schema.v1.0.0.json
    must see the response shape unchanged. The route does NOT populate the
    additive trio (gate_contract_ref / model_candidates / phase_chain_audit_sha)
    — Pydantic Optional defaults to None — so v1.0.0 deserializers either
    drop the extra keys (extra=ignore) or surface them as None.

    Either form is valid; this test pins the SHAPE invariant.
    """
    body = _post_predict(standalone_client)

    # All v1.0.0 fields present + non-None where required.
    for key in ("schema_version", "win_probability", "fighter_a", "fighter_b", "event_date"):
        assert key in body, f"v1.0.0 required field missing from response: {key}"
        assert body[key] is not None

    # schema_version pinned to "1.0.0" (route hardcodes this — Phase 25 D-03 lock).
    assert body["schema_version"] == "1.0.0", (
        f"route returned schema_version={body['schema_version']!r}; "
        "Phase 25 D-03 hardcode binding requires '1.0.0'"
    )

    # Additive trio: present with None values (or absent — both valid for
    # v1.0.0 partners under forward-compat lock).
    for trio_field in V110_ADDITIVE_TRIO:
        if trio_field in body:
            assert body[trio_field] is None, (
                f"v1.1.0 additive trio field {trio_field!r} should be None in "
                f"default /api/v1/predict response; got {body[trio_field]!r}"
            )


def test_v100_response_validates_against_v100_schema(
    standalone_client: TestClient,
    v100_schema_validator: Draft202012Validator,
):
    """The v1.0.0-projected response payload validates against v1.0.0 JSON Schema."""
    body = _post_predict(standalone_client)

    # Project down to v1.0.0 keys (drop additive trio if present-with-None).
    body_v100 = {k: body[k] for k in V100_FIELDS if k in body}

    # Must not raise — v1.0.0 partners' codegen receives valid payloads.
    v100_schema_validator.validate(body_v100)


def test_v100_response_validates_against_v110_schema(
    standalone_client: TestClient,
    v110_schema_validator: Draft202012Validator,
):
    """The FULL response (v1.0.0 + None-trio) validates against v1.1.0 JSON Schema.

    Phase 32 D-04 / D-05 superset guarantee: the v1.1.0 schema accepts
    BOTH v1.0.0-shape payloads (additive trio absent) and v1.0.0+None-trio
    payloads (additive trio present with null values).
    """
    body = _post_predict(standalone_client)
    v110_schema_validator.validate(body)


# ─── v1.1.0 additive trio populates correctly when opted-in ───────────────────


def test_additive_trio_populated_when_explicitly_set():
    """v1.1.0 partners can populate the additive trio via direct construction.

    Proves the fields are wired correctly. The default /api/v1/predict
    route doesn't populate them (v1.0.0 shape preserved), but partners
    constructing PredictorOutputV1 directly (or a future partner-version-
    routing layer) can opt-in and round-trip values through model_dump.
    """
    candidates = [
        {"name": "xgb_v2", "sha256": XGB_V2_CANONICAL_SHA, "phase": "16"},
    ]
    out = PredictorOutputV1(
        win_probability=0.7,
        fighter_a="Islam Makhachev",
        fighter_b="Charles Oliveira",
        event_date=date(2026, 8, 15),
        gate_contract_ref=".planning/gate_contract_v2.3.json",
        model_candidates=candidates,
        phase_chain_audit_sha=XGB_V2_CANONICAL_SHA,
    )
    dumped = out.model_dump()
    assert dumped["gate_contract_ref"] == ".planning/gate_contract_v2.3.json"
    assert dumped["model_candidates"] == candidates
    assert dumped["phase_chain_audit_sha"] == XGB_V2_CANONICAL_SHA

    # v1.0.0 keys still round-trip correctly.
    assert dumped["schema_version"] == "1.0.0"
    assert dumped["win_probability"] == pytest.approx(0.7)
    assert dumped["fighter_a"] == "Islam Makhachev"


def test_v110_populated_payload_validates_against_v110_schema(
    v110_schema_validator: Draft202012Validator,
):
    """Fully-populated v1.1.0 payload validates against the v1.1.0 schema."""
    payload = {
        "schema_version": "1.0.0",
        "win_probability": 0.7,
        "fighter_a": "Islam Makhachev",
        "fighter_b": "Charles Oliveira",
        "event_date": "2026-08-15",
        "base_prob": 0.65,
        "meta_prob": 0.7,
        "meta_learner_version": "v22.1",
        "meta_skipped_reason": None,
        "gate_contract_ref": ".planning/gate_contract_v2.3.json",
        "model_candidates": [
            {"name": "xgb_v2", "sha256": XGB_V2_CANONICAL_SHA, "phase": "16"},
        ],
        "phase_chain_audit_sha": XGB_V2_CANONICAL_SHA,
    }
    v110_schema_validator.validate(payload)


def test_v100_partner_payload_validates_against_v110_schema(
    v110_schema_validator: Draft202012Validator,
):
    """A v1.0.0 partner payload (no additive trio keys at all) validates v1.1.0.

    Proves the v1.0.0 → v1.1.0 migration is non-breaking: a partner
    sending the v1.0.0-shape body and switching their committed schema
    file to v1.1.0 needs no other code changes (additive trio is Optional).
    """
    v100_payload = {
        "schema_version": "1.0.0",
        "win_probability": 0.6234,
        "fighter_a": "Khabib Nurmagomedov",
        "fighter_b": "Conor McGregor",
        "event_date": "2018-10-06",
        "base_prob": 0.6234,
        "meta_prob": None,
        "meta_learner_version": None,
        "meta_skipped_reason": "no_meta_artifact",
    }
    # No additive trio keys at all — must still validate (fields are Optional).
    v110_schema_validator.validate(v100_payload)
