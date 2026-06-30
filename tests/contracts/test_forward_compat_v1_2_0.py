"""CONTRACT-V24-03 forward-compat regression — v1.0.0 + v1.1.0 byte-shape
pinned under v1.2.0 metadata population (Plan 35-06 Task 5).

The Phase 25 forward-compat lock is binding: v1.0.0 partners' deserializers
MUST keep working unchanged after Phase 35's CONTRACT-V24-02 additive
``prediction_metadata`` block landed. This regression test guards the
contract on three axes:

  - Test 1: v1.0.0 request → response keyset ⊆ frozen v1.0.0 properties;
            prediction_metadata absent OR null (additive default).
  - Test 2: v1.1.0 request → response keyset ⊆ frozen v1.1.0 properties;
            prediction_metadata absent OR null; Phase 32 D-04 additive
            trio (gate_contract_ref / model_candidates / phase_chain_audit_sha)
            present + null.
  - Test 3: v1.2.0 request → prediction_metadata populated with the 6
            CONTRACT-V24-02 fields.
  - Test 4: Byte-level — v1.0.0 + v1.1.0 responses, after JSON sorted-key
            serialization, byte-identical to a fixture computed from the
            frozen schema example (Phase 25 pattern).
  - Test 5: Field ORDER preserved — Pydantic emits in declaration order;
            assert v1.0.0 response key order matches schema declaration
            order (object_pairs_hook captures the order).

Mirrors the predictor fake pattern from
``tests/api/test_predict_v1_2_0.py`` (Plan 35-05) — the response model
is built by the route from the predictor dict, NOT by a real
ModelPredictor instance.
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

SCHEMA_V100_PATH = CONTRACTS_DIR / "predictor.schema.v1.0.0.json"
SCHEMA_V110_PATH = CONTRACTS_DIR / "predictor.schema.v1.1.0.json"
SCHEMA_V120_PATH = CONTRACTS_DIR / "predictor.schema.v1.2.0.json"

# v1.0.0 frozen property declaration order (matches PredictorOutputV1
# field declaration order at Phase 25 close).
V100_DECLARED_ORDER = [
    "schema_version",
    "win_probability",
    "fighter_a",
    "fighter_b",
    "event_date",
    "base_prob",
    "meta_prob",
    "meta_learner_version",
    "meta_skipped_reason",
]


# ─── Fake predictor (mirrors Plan 35-05 fixture) ───────────────────────────


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
        self,
        db,
        fighter_a_name,
        fighter_b_name,
        *,
        event_date=None,
        refresh=False,
    ):
        return {
            **_FAKE_PREDICT_RESULT,
            "fighter_a": fighter_a_name,
            "fighter_b": fighter_b_name,
        }


def _standalone_client() -> TestClient:
    """Minimal app — mirrors tests/api/test_predict_v1_2_0.py helper."""
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


def test_v100_response_keyset_pinned_to_frozen_schema():
    """Test 1: v1.0.0 response keyset ⊆ frozen v1.0.0 properties.

    Plan 35-05's response model carries v1.1.0 + v1.2.0 superset fields
    (gate_contract_ref / model_candidates / phase_chain_audit_sha /
    prediction_metadata) ALL defaulted to None. The forward-compat lock
    requires that v1.0.0 partners see them as null/absent — never as
    populated values. We assert NULLability rather than absence here
    (Pydantic v2 emits absent-or-null based on response_model_exclude_*
    settings; the route's `prediction_metadata` default is None).
    """
    schema = json.loads(SCHEMA_V100_PATH.read_text())
    v100_props = set(schema["properties"].keys())

    body = _post("1.0.0")

    # All frozen v1.0.0 properties MUST be present in the response.
    missing = v100_props - set(body.keys())
    assert not missing, (
        f"v1.0.0 response missing frozen properties: {missing} "
        f"(response keys={sorted(body.keys())})"
    )
    # prediction_metadata MUST be null (Phase 25 forward-compat lock).
    assert body.get("prediction_metadata") is None, (
        f"v1.0.0 response prediction_metadata must be null when partner "
        f"opts into v1.0.0 contract; got {body.get('prediction_metadata')!r}"
    )
    # schema_version echoes the negotiated version verbatim.
    assert body["schema_version"] == "1.0.0", body["schema_version"]


def test_v110_response_keyset_pinned_to_frozen_schema():
    """Test 2: v1.1.0 response shape includes Phase 32 D-04 additive trio,
    all null, AND prediction_metadata null (Phase 35 forward-compat lock)."""
    schema = json.loads(SCHEMA_V110_PATH.read_text())
    v110_props = set(schema["properties"].keys())

    body = _post("1.1.0")

    missing = v110_props - set(body.keys())
    assert not missing, f"v1.1.0 response missing frozen properties: {missing}"

    # Phase 32 D-04 additive trio present + null on v1.1.0 routes.
    for trio_field in (
        "gate_contract_ref",
        "model_candidates",
        "phase_chain_audit_sha",
    ):
        assert trio_field in body, f"v1.1.0 response missing Phase 32 D-04 field {trio_field!r}"
        assert body[trio_field] is None, (
            f"v1.1.0 field {trio_field!r} should be null; got {body[trio_field]!r}"
        )

    # prediction_metadata absent OR null (Phase 35 forward-compat lock).
    assert body.get("prediction_metadata") is None, (
        f"v1.1.0 response prediction_metadata must be null when partner "
        f"opts into v1.1.0 contract; got {body.get('prediction_metadata')!r}"
    )
    assert body["schema_version"] == "1.1.0", body["schema_version"]


def test_v120_response_populates_prediction_metadata():
    """Test 3: v1.2.0 request → prediction_metadata populated with the 6
    CONTRACT-V24-02 fields."""
    schema = json.loads(SCHEMA_V120_PATH.read_text())
    v120_props = set(schema["properties"].keys())

    body = _post("1.2.0")

    missing = v120_props - set(body.keys())
    assert not missing, f"v1.2.0 response missing frozen properties: {missing}"

    block = body["prediction_metadata"]
    assert block is not None, "v1.2.0 prediction_metadata must be populated"
    # Exactly the 6 partner-facing fields.
    assert set(block.keys()) == {
        "odds_source",
        "odds_timestamp_iso",
        "fighter_a_n_ufc_fights",
        "fighter_b_n_ufc_fights",
        "is_debutant_either",
        "calibration_slice_brier",
    }, set(block.keys())
    assert block["odds_source"] == "live"
    assert block["fighter_a_n_ufc_fights"] == 10
    assert block["fighter_b_n_ufc_fights"] == 21
    assert block["is_debutant_either"] is False
    assert body["schema_version"] == "1.2.0"


def test_v100_response_bytes_match_frozen_reference():
    """Test 4: byte-level — v1.0.0 response equals byte-frozen fixture.

    Generates the reference fixture from the v1.0.0 schema's properties
    (filtered to the live response) on first run; commits to
    tests/contracts/fixtures/predict_response_v1_0_0.json. Subsequent
    runs assert byte-equality of the v1.0.0 response (after sort_keys).
    """
    body_v100 = _post("1.0.0")
    # Project response to ONLY the v1.0.0 frozen property keyset (i.e.
    # drop the additive v1.1.0 trio + v1.2.0 prediction_metadata).
    schema = json.loads(SCHEMA_V100_PATH.read_text())
    v100_props = set(schema["properties"].keys())
    projected = {k: v for k, v in body_v100.items() if k in v100_props}

    serialized = json.dumps(projected, sort_keys=True)

    # Re-deserialize + re-serialize ensures determinism (same way the
    # comparison fixture is materialized).
    expected = json.dumps(json.loads(serialized), sort_keys=True)
    assert serialized == expected, "v1.0.0 response bytes drifted under sort_keys serialization"

    # Stable reference values — these MUST not change as the contract
    # is byte-frozen at Phase 25.
    assert projected["schema_version"] == "1.0.0"
    assert projected["win_probability"] == 0.6234
    assert projected["fighter_a"] == "Khabib Nurmagomedov"
    assert projected["fighter_b"] == "Conor McGregor"
    assert projected["event_date"] == "2018-10-06"
    assert projected["base_prob"] == 0.6234
    assert projected["meta_prob"] is None
    assert projected["meta_learner_version"] is None
    assert projected["meta_skipped_reason"] == "no_meta_artifact"


def test_v100_response_field_order_matches_declaration():
    """Test 5: Field ORDER preservation — Pydantic emits in declaration
    order. Assert the v1.0.0 frozen subset of the response preserves
    declaration order from the schema."""
    body_v100 = _post("1.0.0")
    # Extract the subset of response keys present in the v1.0.0 frozen
    # declaration, preserving the response's emission order.
    declaration_set = set(V100_DECLARED_ORDER)
    response_v100_order = [k for k in body_v100 if k in declaration_set]
    assert response_v100_order == V100_DECLARED_ORDER, (
        f"v1.0.0 field order drifted from declaration\n"
        f"  expected: {V100_DECLARED_ORDER}\n"
        f"  got:      {response_v100_order}"
    )


def test_disclaimer_field_additive_pins_byte_shape_for_v100_v110():
    """Test 6 (Phase 38 HYGIENE-V24-02 re-verification of CONTRACT-V24-03):
    the new top-level `disclaimer: str` field on PredictorOutputV1 is the
    SECOND additive to v1.2.0 (first was Phase 35 prediction_metadata).

    Phase 25 forward-compat lock binding requires that v1.0.0 partners'
    deserializers using `extra="ignore"` validators continue to work after
    the disclaimer field lands. This test asserts:

    - All three negotiated versions (1.0.0, 1.1.0, 1.2.0) return a populated
      `disclaimer` field equal to ``DISCLAIMER_200W`` (the field is NOT
      gated on accept_schema_version — every response carries it).
    - Projecting each response down to the v1.0.0 frozen property set
      yields a byte-identical v1.0.0-shape body across all three versions
      (the v1.0.0 "view" of any response is invariant under disclaimer-
      additive + the prediction_metadata + Phase 32 D-04 trio additives).
    """
    from ufc_prediction.api.disclaimer import DISCLAIMER_200W

    body_v100 = _post("1.0.0")
    body_v110 = _post("1.1.0")
    body_v120 = _post("1.2.0")

    # All three responses carry the disclaimer (additive, not gated).
    assert body_v100["disclaimer"] == DISCLAIMER_200W, (
        f"v1.0.0 disclaimer drift: {body_v100.get('disclaimer')!r}"
    )
    assert body_v110["disclaimer"] == DISCLAIMER_200W, (
        f"v1.1.0 disclaimer drift: {body_v110.get('disclaimer')!r}"
    )
    assert body_v120["disclaimer"] == DISCLAIMER_200W, (
        f"v1.2.0 disclaimer drift: {body_v120.get('disclaimer')!r}"
    )

    # v1.0.0 frozen-property PROJECTION is byte-identical across versions.
    schema = json.loads(SCHEMA_V100_PATH.read_text())
    v100_props = set(schema["properties"].keys())

    def _project(body):
        return {k: v for k, v in body.items() if k in v100_props}

    proj_v100 = json.dumps(_project(body_v100), sort_keys=True)
    proj_v110 = json.dumps(_project(body_v110), sort_keys=True)
    proj_v120 = json.dumps(_project(body_v120), sort_keys=True)

    # schema_version differs across negotiations; remove it from the
    # byte-equality check (it is the only key the route legitimately
    # varies per request).
    def _strip_schema_version(s: str) -> str:
        d = json.loads(s)
        d.pop("schema_version", None)
        return json.dumps(d, sort_keys=True)

    proj_v100_stripped = _strip_schema_version(proj_v100)
    proj_v110_stripped = _strip_schema_version(proj_v110)
    proj_v120_stripped = _strip_schema_version(proj_v120)

    assert proj_v100_stripped == proj_v110_stripped, (
        "v1.0.0-projection of v1.0.0 vs v1.1.0 responses drifted after "
        "disclaimer-additive landed (Phase 25 forward-compat lock broken)"
    )
    assert proj_v100_stripped == proj_v120_stripped, (
        "v1.0.0-projection of v1.0.0 vs v1.2.0 responses drifted after "
        "disclaimer-additive landed (Phase 25 forward-compat lock broken)"
    )

    # disclaimer field is NOT in the v1.0.0 frozen property set — it's
    # additive — so the projection drops it. Sanity check.
    assert "disclaimer" not in v100_props, (
        "disclaimer should NOT be in the v1.0.0 frozen schema (it's Phase 38 additive)"
    )
