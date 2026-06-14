"""Phase 25 + Phase 32 + Phase 35 + Phase 69 drift sentinel — committed artifacts MUST match live emission.

Per Pitfall #3 (schema-emission drift): if PredictorOutputV1 changes but
the appropriate predictor.schema.vX.Y.Z.json is NOT re-emitted, this
test fails LOUDLY.
Per Pitfall #4 (non-deterministic ordering): comparison is structural
(dict-eq) not text-diff — key ordering noise does not trigger false failures.

Phase 32 D-04 extension (PARTNER-V23-01):
  - v1.0.0 schema FILE is byte-frozen (Phase 25 forward-compat lock binding).
    Drift check is against a hand-curated property set rather than live
    Pydantic emission (live model now emits v1.3.0 shape post-Phase-69).
  - v1.1.0 schema + openapi files are BYTE-FROZEN (Phase 35 extension — Plan
    35-06 owns the v1.1.0 freeze; v1.2.0 was live until Phase 69).
  - v1.2.0 schema + openapi files are BYTE-FROZEN (Phase 69 API-V261-01
    extension — v1.3.0 is now the live source-of-truth).
  - v1.3.0 schema + openapi files are the NEW source-of-truth for live
    Pydantic emission. These tests are the gate that catches schema drift
    the moment Phase 70+ touches PredictorOutputV1.
"""
from __future__ import annotations

import json
from pathlib import Path

from ufc_prediction.api.app import create_app
from ufc_prediction.api.v1.models import PredictorOutputV1

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONTRACTS_DIR = REPO_ROOT / "src" / "ufc_prediction" / "contracts"

# Phase 25 byte-frozen artifacts (Phase 25 forward-compat lock binding).
SCHEMA_V100_PATH = CONTRACTS_DIR / "predictor.schema.v1.0.0.json"
OPENAPI_V100_PATH = CONTRACTS_DIR / "openapi.v1.0.0.json"

# Phase 32 D-04 artifacts (BYTE-FROZEN as of Phase 35 — Plan 35-06).
SCHEMA_V110_PATH = CONTRACTS_DIR / "predictor.schema.v1.1.0.json"
OPENAPI_V110_PATH = CONTRACTS_DIR / "openapi.v1.1.0.json"

# Phase 35 CONTRACT-V24-02 sibling artifacts (BYTE-FROZEN as of Phase 69).
SCHEMA_V120_PATH = CONTRACTS_DIR / "predictor.schema.v1.2.0.json"
OPENAPI_V120_PATH = CONTRACTS_DIR / "openapi.v1.2.0.json"

# Phase 69 API-V261-01 sibling artifacts (live Pydantic source-of-truth).
SCHEMA_V130_PATH = CONTRACTS_DIR / "predictor.schema.v1.3.0.json"
OPENAPI_V130_PATH = CONTRACTS_DIR / "openapi.v1.3.0.json"

# Hand-curated v1.0.0 property set (Phase 25 lock — must NEVER drift).
EXPECTED_V100_PROPERTIES: set[str] = {
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

# Hand-curated v1.1.0 property set (Phase 32 D-04 + Phase 35 byte-freeze).
# v1.0.0 (9 fields) + Phase 32 additive trio (3 fields) = 12 fields.
EXPECTED_V110_PROPERTIES: set[str] = EXPECTED_V100_PROPERTIES | {
    "gate_contract_ref",
    "model_candidates",
    "phase_chain_audit_sha",
}

# Hand-curated v1.2.0 property set (Phase 35 + 38 + Phase 69 byte-freeze).
# v1.1.0 (12 fields) + prediction_metadata (Phase 35) + disclaimer (Phase 38) = 14 fields.
EXPECTED_V120_PROPERTIES: set[str] = EXPECTED_V110_PROPERTIES | {
    "prediction_metadata",
    "disclaimer",
}


# ─── v1.0.0 byte-frozen preservation (Phase 25 lock) ──────────────────────────


def test_predictor_schema_v100_preserved():
    """v1.0.0 schema FILE is byte-frozen — Phase 25 forward-compat lock.

    Asserts the committed file still carries exactly the 9 Phase 25 fields
    (no additions, no removals). Phase 32+ MUST NOT overwrite this file —
    the live Pydantic model now emits a SUPERSET (v1.2.0 = v1.1.0 + 1
    additive Optional field). v1.1.0 + v1.2.0 live in sibling files.
    """
    committed = json.loads(SCHEMA_V100_PATH.read_text())
    props = set(committed["properties"].keys())
    assert props == EXPECTED_V100_PROPERTIES, (
        f"v1.0.0 schema FILE drifted (Phase 25 forward-compat lock violation!): "
        f"extras={props - EXPECTED_V100_PROPERTIES} "
        f"missing={EXPECTED_V100_PROPERTIES - props}"
    )
    assert len(committed["properties"]) == 9, (
        f"v1.0.0 schema must carry exactly 9 properties; found {len(committed['properties'])}"
    )
    # required field list is also locked
    assert set(committed["required"]) == {
        "win_probability", "fighter_a", "fighter_b", "event_date",
    }, f"v1.0.0 required list drifted: {committed['required']}"


def test_openapi_v100_preserved():
    """v1.0.0 OpenAPI FILE is byte-frozen — Phase 25 forward-compat lock."""
    committed = json.loads(OPENAPI_V100_PATH.read_text())
    assert committed["openapi"] == "3.1.0", (
        f"v1.0.0 openapi version drifted: {committed['openapi']}"
    )
    assert "/api/v1/predict" in committed["paths"], (
        "v1.0.0 openapi file lost /api/v1/predict path (Phase 25 lock violation)"
    )


# ─── v1.1.0 byte-frozen preservation (Phase 35 extends Phase 32 lock) ─────────


def test_predictor_schema_v110_preserved():
    """v1.1.0 schema FILE is byte-frozen — Phase 35 forward-compat lock extension.

    Asserts the committed file still carries exactly the 12 Phase 32 D-04
    fields (v1.0.0 9 + additive trio 3). Phase 35+ MUST NOT overwrite this
    file — live Pydantic now emits v1.2.0 shape. v1.1.0 partner contract
    is frozen.
    """
    committed = json.loads(SCHEMA_V110_PATH.read_text())
    props = set(committed["properties"].keys())
    assert props == EXPECTED_V110_PROPERTIES, (
        f"v1.1.0 schema FILE drifted (Phase 35 forward-compat lock violation!): "
        f"extras={props - EXPECTED_V110_PROPERTIES} "
        f"missing={EXPECTED_V110_PROPERTIES - props}"
    )
    assert len(committed["properties"]) == 12, (
        f"v1.1.0 schema must carry exactly 12 properties; found {len(committed['properties'])}"
    )
    # required field list preserved verbatim from v1.0.0 (additive trio is
    # all Optional).
    assert set(committed["required"]) == {
        "win_probability", "fighter_a", "fighter_b", "event_date",
    }, f"v1.1.0 required list drifted: {committed['required']}"


def test_openapi_v110_preserved():
    """v1.1.0 OpenAPI FILE is byte-frozen — Phase 35 forward-compat lock extension."""
    committed = json.loads(OPENAPI_V110_PATH.read_text())
    assert committed["openapi"] == "3.1.0", (
        f"v1.1.0 openapi version drifted: {committed['openapi']}"
    )
    assert "/api/v1/predict" in committed["paths"], (
        "v1.1.0 openapi file lost /api/v1/predict path (Phase 35 lock violation)"
    )


# ─── v1.2.0 byte-frozen preservation (Phase 69 extends Phase 35 lock) ─────────


def test_predictor_schema_v120_preserved():
    """v1.2.0 schema FILE is byte-frozen — Phase 69 forward-compat lock extension.

    Asserts the committed file still carries exactly the 14 properties
    (v1.0.0 9 + Phase 32 trio 3 + prediction_metadata + disclaimer). Phase
    69+ MUST NOT overwrite this file — live Pydantic now emits v1.3.0 shape.
    """
    committed = json.loads(SCHEMA_V120_PATH.read_text())
    props = set(committed["properties"].keys())
    assert props == EXPECTED_V120_PROPERTIES, (
        f"v1.2.0 schema FILE drifted (Phase 69 forward-compat lock violation!): "
        f"extras={props - EXPECTED_V120_PROPERTIES} "
        f"missing={EXPECTED_V120_PROPERTIES - props}"
    )
    assert len(committed["properties"]) == 14, (
        f"v1.2.0 schema must carry exactly 14 properties; "
        f"found {len(committed['properties'])}"
    )
    # required field list preserved verbatim from v1.0.0/v1.1.0.
    assert set(committed["required"]) == {
        "win_probability", "fighter_a", "fighter_b", "event_date",
    }, f"v1.2.0 required list drifted: {committed['required']}"


def test_openapi_v120_preserved():
    """v1.2.0 OpenAPI FILE is byte-frozen — Phase 69 forward-compat lock extension."""
    committed = json.loads(OPENAPI_V120_PATH.read_text())
    assert committed["openapi"] == "3.1.0", (
        f"v1.2.0 openapi version drifted: {committed['openapi']}"
    )
    assert "/api/v1/predict" in committed["paths"], (
        "v1.2.0 openapi file lost /api/v1/predict path (Phase 69 lock violation)"
    )


# ─── v1.3.0 live emission match (Phase 69 API-V261-01 drift sentinel) ─────────


def test_predictor_schema_v130_matches_live_emission():
    """PredictorOutputV1.model_json_schema() byte-equals predictor.schema.v1.3.0.json.

    Phase 69 API-V261-01: live Pydantic is the v1.3.0 source-of-truth.
    If PredictorOutputV1 changes (field added/removed/retyped) AND
    `scripts/emit_partner_contracts.py` is not re-run, this test fails
    LOUDLY.
    """
    committed = json.loads(SCHEMA_V130_PATH.read_text())
    live = PredictorOutputV1.model_json_schema()
    assert live == committed, (
        "PredictorOutputV1 changed but predictor.schema.v1.3.0.json was "
        "not re-emitted. Either: (a) bump version (v1.3.0 → v1.4.0) and "
        "re-run `python scripts/emit_partner_contracts.py`, OR "
        "(b) revert the PredictorOutputV1 change."
    )


def test_openapi_v130_matches_live_emission():
    """create_app().openapi() byte-equals committed openapi.v1.3.0.json.

    Phase 69 API-V261-01 sibling artifact; mirrors v1.0.0 + v1.1.0 + v1.2.0
    drift sentinel pattern. Captures the live FastAPI app surface
    including all endpoints mounted as of Phase 69 close.
    """
    committed = json.loads(OPENAPI_V130_PATH.read_text())
    live = create_app().openapi()
    assert live == committed, (
        "FastAPI app surface changed but openapi.v1.3.0.json was not "
        "re-emitted. Re-run `python scripts/emit_partner_contracts.py`."
    )


# ─── v1.1.0 additive-only assertion (Phase 25 lock binding) ───────────────────


def test_v110_additive_only_over_v100():
    """v1.1.0 schema = v1.0.0 ∪ {gate_contract_ref, model_candidates, phase_chain_audit_sha}.

    The whole point of Phase 32 D-04 — v1.0.0 partners' deserializers
    must keep working unchanged. Concretely:
      - No v1.0.0 property has been removed or renamed in v1.1.0.
      - The only new properties are the additive trio.
      - All 3 new properties default to null (forward-compat lock binding).
    """
    v100 = json.loads(SCHEMA_V100_PATH.read_text())
    v110 = json.loads(SCHEMA_V110_PATH.read_text())

    v100_props = set(v100["properties"].keys())
    v110_props = set(v110["properties"].keys())

    removed = v100_props - v110_props
    assert removed == set(), f"v1.1.0 REMOVED v1.0.0 fields: {removed}"

    added = v110_props - v100_props
    assert added == {"gate_contract_ref", "model_candidates", "phase_chain_audit_sha"}, (
        f"v1.1.0 added fields = {added}; expected the additive trio"
    )

    # All 3 additive properties default to null.
    for field in added:
        default = v110["properties"][field].get("default")
        assert default is None, (
            f"v1.1.0 field {field!r} default = {default!r}; must be null "
            "(Phase 25 forward-compat lock binding)"
        )

    # v1.0.0 required set is preserved verbatim — no new required fields.
    assert set(v110["required"]) == set(v100["required"]), (
        f"v1.1.0 required list drifted from v1.0.0: "
        f"v1.0.0={v100['required']} v1.1.0={v110['required']}"
    )


# ─── v1.2.0 additive-only assertion (Phase 35 CONTRACT-V24-03 binding) ────────


def test_v120_additive_only_over_v110():
    """v1.2.0 schema = v1.1.0 ∪ {prediction_metadata, disclaimer}.

    Phase 35 CONTRACT-V24-03 forward-compat lock — v1.1.0 partners'
    deserializers must keep working unchanged. Phase 38 HYGIENE-V24-02
    extends v1.2.0 with a second additive (`disclaimer`); both are
    additive-only over v1.1.0. Concretely:
      - No v1.1.0 property has been removed or renamed in v1.2.0.
      - The only new properties are `prediction_metadata` (Phase 35)
        and `disclaimer` (Phase 38).
      - `prediction_metadata` defaults to null (Phase 25/35 lock).
      - `disclaimer` defaults to the DISCLAIMER_200W constant string
        (regulatory surface — non-null by design; additivity preserved
        via default-value mechanism + extra='ignore' on partner side).
    """
    v110 = json.loads(SCHEMA_V110_PATH.read_text())
    v120 = json.loads(SCHEMA_V120_PATH.read_text())

    v110_props = set(v110["properties"].keys())
    v120_props = set(v120["properties"].keys())

    removed = v110_props - v120_props
    assert removed == set(), f"v1.2.0 REMOVED v1.1.0 fields: {removed}"

    added = v120_props - v110_props
    assert added == {"prediction_metadata", "disclaimer"}, (
        f"v1.2.0 added fields = {added}; expected "
        f"{{prediction_metadata, disclaimer}}"
    )

    # prediction_metadata defaults to null (Phase 25/35 lock binding).
    pm_default = v120["properties"]["prediction_metadata"].get("default")
    assert pm_default is None, (
        f"v1.2.0 field 'prediction_metadata' default = {pm_default!r}; "
        "must be null (Phase 35 forward-compat lock binding)"
    )

    # disclaimer defaults to the DISCLAIMER_200W string constant
    # (Phase 38 HYGIENE-V24-02 — regulatory surface; non-null default
    # but additive because v1.0.0/v1.1.0 partners use extra='ignore').
    from ufc_prediction.api.disclaimer import DISCLAIMER_200W
    disc_default = v120["properties"]["disclaimer"].get("default")
    assert disc_default == DISCLAIMER_200W, (
        f"v1.2.0 field 'disclaimer' default drift; expected DISCLAIMER_200W"
    )

    # v1.1.0 required set is preserved verbatim — no new required fields.
    assert set(v120["required"]) == set(v110["required"]), (
        f"v1.2.0 required list drifted from v1.1.0: "
        f"v1.1.0={v110['required']} v1.2.0={v120['required']}"
    )


# ─── v1.3.0 no-additive assertion (Phase 69 API-V261-01 binding) ──────────────


def test_v130_no_additions_over_v120():
    """v1.3.0 schema = v1.2.0 schema (no success-path additives).

    Phase 69 API-V261-01 binding: v1.3.0 success-path bytes IDENTICAL
    to v1.2.0 by design. The v1.3.0 capability is the RFC 7807 error
    wrapper (Phase 52 API-V26-01), which lives on the error path and is
    opt-in via the Accept header. NO success-path schema additive shipped.

    Concretely:
      - Property set is identical (no fields added, removed, renamed).
      - required list is identical.
      - Per-property defaults are identical.

    Pydantic JSON-Schema serialization metadata (e.g.,
    `additionalProperties` representation for `model_candidates.items`)
    MAY differ between the committed v1.2.0 + v1.3.0 files — that's a
    Pydantic-version drift across when each was emitted, not a contract
    change. We pin the partner-visible surface here; byte-level drift
    against live emission is owned by
    `test_predictor_schema_v130_matches_live_emission`.
    """
    v120 = json.loads(SCHEMA_V120_PATH.read_text())
    v130 = json.loads(SCHEMA_V130_PATH.read_text())

    v120_props = set(v120["properties"].keys())
    v130_props = set(v130["properties"].keys())

    removed = v120_props - v130_props
    assert removed == set(), f"v1.3.0 REMOVED v1.2.0 fields: {removed}"

    added = v130_props - v120_props
    assert added == set(), (
        f"v1.3.0 added fields over v1.2.0 = {added}; expected NONE. "
        "Phase 69 API-V261-01 binding: v1.3.0 success-path bytes IDENTICAL "
        "to v1.2.0. New success-path fields require a v1.4.0 bump."
    )

    # Per-property defaults preserved verbatim.
    for prop in v120_props:
        d_v120 = v120["properties"][prop].get("default")
        d_v130 = v130["properties"][prop].get("default")
        assert d_v120 == d_v130, (
            f"v1.3.0 property {prop!r} default DRIFTED from v1.2.0:\n"
            f"  v1.2.0: {d_v120!r}\n"
            f"  v1.3.0: {d_v130!r}"
        )

    # v1.2.0 required set preserved verbatim.
    assert set(v130["required"]) == set(v120["required"]), (
        f"v1.3.0 required list drifted from v1.2.0: "
        f"v1.2.0={v120['required']} v1.3.0={v130['required']}"
    )
