"""API-V261-02 forward-compat regression — v1.3.0 success-path bytes pinned.

Phase 69 (v2.6.1) extends the Phase 35 CONTRACT-V24-03 forward-compat
suite to cover the v1.3.0 negotiated version. The v1.3.0 capability
(RFC 7807 application/problem+json error wrapper, Phase 52 API-V26-01)
lives entirely on the ERROR path and is opt-in via the Accept header,
so the success path for v1.3.0 partners is BYTE-IDENTICAL to v1.2.0.

This regression locks two invariants:

  1. Success path: a v1.3.0 request body
     (`accept_schema_version="1.3.0"`) returns a response that is
     byte-identical, key-for-key + value-for-value, to the v1.2.0
     response — except for `schema_version` echoing "1.3.0" vs "1.2.0".
     Reusing the Phase 35 fake predictor + standalone client pattern.

  2. Frozen schema files: the committed v1.3.0 + v1.2.0 schema files
     have the same property set (no fields added, removed, or retyped
     on the response). Pydantic JSON-Schema serialization metadata may
     differ between the two committed files (e.g.,
     `additionalProperties` representation for `model_candidates`); we
     pin the partner-visible semantic surface, not the JSON-Schema
     metadata. See `tests/contracts/test_no_drift.py` for the
     byte-level drift sentinel on v1.3.0 against the live emission.

Test pattern mirrors `tests/contracts/test_forward_compat_v1_2_0.py`
exactly — same fake predictor, same standalone client, same
`_post(version)` helper — so the v1.3.0 regression composes with the
existing Phase 35 lock and shares the same false-positive surface.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ufc_prediction.api.deps import get_db
from ufc_prediction.api.v1 import predict as v1_predict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONTRACTS_DIR = REPO_ROOT / "src" / "ufc_prediction" / "contracts"

SCHEMA_V120_PATH = CONTRACTS_DIR / "predictor.schema.v1.2.0.json"
SCHEMA_V130_PATH = CONTRACTS_DIR / "predictor.schema.v1.3.0.json"


# ─── Fake predictor (mirrors Phase 35 + Phase 52 fixture) ───────────────────


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
    "odds_source": "live",
    "prediction_metadata": {
        "win_probability_source": "xgb_v2_no_odds",
        "odds_source": "live",
        "odds_timestamp_iso": "2018-10-06T22:00:00+00:00",
        "fighter_a_n_ufc_fights": 10,
        "fighter_b_n_ufc_fights": 21,
        "is_debutant_either": False,
        "calibration_slice_brier": 0.19,
    },
}


class _FakePredictor:
    """Stable predict() that always returns the canonical Khabib/Conor body."""

    def predict(
        self, db, fighter_a_name, fighter_b_name, *, event_date=None, refresh=False,
    ):
        return {
            **_FAKE_PREDICT_RESULT,
            "fighter_a": fighter_a_name,
            "fighter_b": fighter_b_name,
        }


def _standalone_client() -> TestClient:
    """Minimal app — mirrors Phase 35 / Phase 52 helper."""
    app = FastAPI()
    app.include_router(v1_predict.router, prefix="/api/v1")

    def _fake_db():
        yield None

    app.dependency_overrides[get_db] = _fake_db
    return TestClient(app)


def _post(version: str) -> dict:
    """POST /api/v1/predict with accept_schema_version=version; return body."""
    body = {
        "fighter_a": "Khabib Nurmagomedov",
        "fighter_b": "Conor McGregor",
        "event_date": "2018-10-06",
        "accept_schema_version": version,
    }
    with patch(
        "ufc_prediction.api.v1.predict._get_predictor",
        return_value=_FakePredictor(),
    ):
        resp = _standalone_client().post("/api/v1/predict", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


# ─── Tests ────────────────────────────────────────────────────────────────


def test_v130_response_keyset_pinned_to_v120_shape():
    """Test 1: v1.3.0 response keyset ⊆ v1.2.0 properties + 'schema_version'.

    The committed v1.2.0 schema is the source-of-truth for the v1.3.0
    response shape (Phase 69 D-01: success path byte-identical). Any
    field present in the v1.3.0 response MUST also appear in the v1.2.0
    schema property set — no v1.3.0-only success-path fields.
    """
    schema_v120 = json.loads(SCHEMA_V120_PATH.read_text())
    v120_props = set(schema_v120["properties"].keys())

    body = _post("1.3.0")

    extras = set(body.keys()) - v120_props
    assert extras == set(), (
        f"v1.3.0 response carries fields NOT in v1.2.0 schema: {extras}. "
        "Phase 69 D-01 binding: v1.3.0 success-path bytes IDENTICAL to "
        "v1.2.0. Any v1.3.0-only field on the success path is a forward-"
        "compat lock violation."
    )

    missing = v120_props - set(body.keys())
    assert not missing, (
        f"v1.3.0 response missing v1.2.0 frozen properties: {missing}"
    )

    assert body["schema_version"] == "1.3.0", body["schema_version"]


def test_v130_populates_prediction_metadata_same_as_v120():
    """Test 2: v1.3.0 prediction_metadata populated identically to v1.2.0.

    The Phase 35 CONTRACT-V24-02 metadata block (6 partner-facing fields)
    is populated for v1.2.0+ partners. v1.3.0 partners get the same
    block — no v1.3.0-only metadata fields.
    """
    body_v120 = _post("1.2.0")
    body_v130 = _post("1.3.0")

    block_v120 = body_v120["prediction_metadata"]
    block_v130 = body_v130["prediction_metadata"]

    assert block_v120 is not None, "v1.2.0 prediction_metadata must be populated"
    assert block_v130 is not None, "v1.3.0 prediction_metadata must be populated"
    assert block_v120 == block_v130, (
        "v1.3.0 prediction_metadata DRIFTED from v1.2.0:\n"
        f"  v1.2.0: {block_v120}\n"
        f"  v1.3.0: {block_v130}"
    )


def test_v130_response_bytes_equal_v120_after_schema_version_strip():
    """Test 3: byte-level — v1.3.0 response == v1.2.0 response after
    stripping the `schema_version` field (the only field that legitimately
    varies per negotiated version).

    This is the load-bearing assertion for the v1.3.0 "byte-identical to
    v1.2.0 by design" invariant. If a future phase adds a v1.3.0-only
    success-path field, this test fails LOUDLY.
    """
    body_v120 = _post("1.2.0")
    body_v130 = _post("1.3.0")

    # Strip schema_version (varies by design).
    body_v120.pop("schema_version", None)
    body_v130.pop("schema_version", None)

    s120 = json.dumps(body_v120, sort_keys=True)
    s130 = json.dumps(body_v130, sort_keys=True)
    assert s120 == s130, (
        "v1.3.0 success-path bytes DRIFTED from v1.2.0 (forward-compat "
        "lock violation — Phase 69 D-01):\n"
        f"  v1.2.0 (sorted): {s120}\n"
        f"  v1.3.0 (sorted): {s130}"
    )


def test_v130_schema_file_properties_match_v120():
    """Test 4: predictor.schema.v1.3.0.json carries the same property set
    as predictor.schema.v1.2.0.json (both are committed, byte-frozen
    siblings under the forward-compat lock).

    Pydantic JSON-Schema serialization metadata (e.g., `additionalProperties`
    representation) MAY differ between the two files — that's a Pydantic
    version drift across when each was emitted, not a contract change.
    We pin the partner-visible surface (property set + required + default
    on each property), not the JSON-Schema implementation detail.
    """
    v120 = json.loads(SCHEMA_V120_PATH.read_text())
    v130 = json.loads(SCHEMA_V130_PATH.read_text())

    v120_props = set(v120["properties"].keys())
    v130_props = set(v130["properties"].keys())

    extras_v130 = v130_props - v120_props
    extras_v120 = v120_props - v130_props
    assert extras_v130 == set(), (
        f"v1.3.0 schema added fields over v1.2.0: {extras_v130}. "
        "Phase 69 D-01 binding: success-path schemas are byte-equivalent."
    )
    assert extras_v120 == set(), (
        f"v1.3.0 schema REMOVED v1.2.0 fields: {extras_v120}. "
        "Forward-compat lock violation."
    )
    assert set(v130["required"]) == set(v120["required"]), (
        f"v1.3.0 required list DRIFTED from v1.2.0:\n"
        f"  v1.2.0: {v120['required']}\n"
        f"  v1.3.0: {v130['required']}"
    )

    # Per-property defaults preserved verbatim (forward-compat lock binding).
    for prop in v120_props:
        d_v120 = v120["properties"][prop].get("default")
        d_v130 = v130["properties"][prop].get("default")
        assert d_v120 == d_v130, (
            f"v1.3.0 property {prop!r} default DRIFTED from v1.2.0:\n"
            f"  v1.2.0: {d_v120!r}\n"
            f"  v1.3.0: {d_v130!r}"
        )


def test_v130_response_carries_disclaimer():
    """Test 5: v1.3.0 response carries the HYGIENE-V24-02 disclaimer
    (Phase 38 additive — non-version-gated).

    Mirrors `test_forward_compat_v1_2_0.py::test_disclaimer_field_additive`
    at the v1.3.0 negotiated version.
    """
    from ufc_prediction.api.disclaimer import DISCLAIMER_200W

    body_v130 = _post("1.3.0")
    assert body_v130["disclaimer"] == DISCLAIMER_200W, (
        f"v1.3.0 disclaimer drift: {body_v130.get('disclaimer')!r}"
    )


def test_v130_does_not_break_older_partners():
    """Test 6: Phase 25 forward-compat lock — v1.0.0 + v1.1.0 + v1.2.0
    partners' success-path bytes UNCHANGED in the presence of the v1.3.0
    Literal extension.

    The Phase 52 API-V26-01 change added "1.3.0" to the
    `accept_schema_version` Literal but MUST NOT alter response bytes
    for older partners. This is the v2.6.1 invariant #3 ("PARTNER schema
    v1.0.0/1.1.0/1.2.0/1.3.0 success-path bytes UNCHANGED") in
    REQUIREMENTS.md.
    """
    body_v100 = _post("1.0.0")
    body_v110 = _post("1.1.0")
    body_v120 = _post("1.2.0")

    assert body_v100["schema_version"] == "1.0.0"
    assert body_v110["schema_version"] == "1.1.0"
    assert body_v120["schema_version"] == "1.2.0"

    # v1.0.0 partners see prediction_metadata=None (forward-compat lock).
    assert body_v100.get("prediction_metadata") is None
    # v1.1.0 partners see prediction_metadata=None (forward-compat lock).
    assert body_v110.get("prediction_metadata") is None
    # v1.2.0 partners see prediction_metadata populated (Phase 35 contract).
    assert body_v120["prediction_metadata"] is not None
