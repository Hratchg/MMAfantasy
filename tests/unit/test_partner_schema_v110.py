"""Phase 32 D-04 unit tests — PARTNER schema v1.1.0 additive trio.

Asserts that PredictorOutputV1 carries 3 additional Optional fields
beyond the v1.0.0 baseline:

  - gate_contract_ref: str | None
  - model_candidates: list[dict] | None
  - phase_chain_audit_sha: str | None

All three default to None, so v1.0.0 partners (Phase 25 forward-compat
lock binding) see the response shape unchanged when they continue to
deserialize against `predictor.schema.v1.0.0.json` (additive fields are
silently dropped or surface as Optional None values).
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from ufc_prediction.api.v1.models import PredictorOutputV1

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCHEMA_V100_PATH = (
    REPO_ROOT / "src" / "ufc_prediction" / "contracts"
    / "predictor.schema.v1.0.0.json"
)

# The exact v1.0.0 field set, hand-curated from Phase 25's locked schema.
# The additive trio MUST NOT touch this set (Phase 25 forward-compat lock).
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

# The 3 Phase 32 additive fields per D-04 (operator-locked at smart-discuss).
EXPECTED_V110_ADDITIVE_TRIO: set[str] = {
    "gate_contract_ref",
    "model_candidates",
    "phase_chain_audit_sha",
}


def test_predictor_output_v110_fields_default_none():
    """v1.1.0 additive trio defaults to None when not explicitly set.

    Instantiates PredictorOutputV1 with only v1.0.0-required fields and
    asserts the additive trio is None — proving v1.0.0 partners see the
    same response shape (Phase 25 D-03 forward-compat lock binding).
    """
    out = PredictorOutputV1(
        win_probability=0.5,
        fighter_a="A",
        fighter_b="B",
        event_date=date.today(),
    )
    assert out.gate_contract_ref is None
    assert out.model_candidates is None
    assert out.phase_chain_audit_sha is None


def test_v110_fields_typing():
    """Pydantic model_fields introspection — the trio is Optional + correctly typed."""
    fields = PredictorOutputV1.model_fields
    assert "gate_contract_ref" in fields
    assert "model_candidates" in fields
    assert "phase_chain_audit_sha" in fields

    # Pydantic v2 annotations are the literal `X | None` Python expressions.
    # Compare via string repr of the annotation (most portable across pydantic versions).
    gcr_ann = fields["gate_contract_ref"].annotation
    assert gcr_ann == (str | None), f"gate_contract_ref annotation = {gcr_ann!r}"

    pca_ann = fields["phase_chain_audit_sha"].annotation
    assert pca_ann == (str | None), f"phase_chain_audit_sha annotation = {pca_ann!r}"

    mc_ann = fields["model_candidates"].annotation
    assert mc_ann == (list[dict] | None), f"model_candidates annotation = {mc_ann!r}"

    # Defaults must be None (Phase 25 forward-compat lock binding).
    assert fields["gate_contract_ref"].default is None
    assert fields["model_candidates"].default is None
    assert fields["phase_chain_audit_sha"].default is None


def test_v110_fields_populate_correctly():
    """v1.1.0 partners can populate the additive trio + round-trip through model_dump."""
    canonical_sha = (
        "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"
    )
    candidates = [
        {"name": "xgb_v2", "sha256": canonical_sha, "phase": "16"},
        {"name": "meta_v2", "sha256": "deadbeef" * 8, "phase": "26"},
    ]
    out = PredictorOutputV1(
        win_probability=0.7,
        fighter_a="Khabib",
        fighter_b="Conor",
        event_date=date(2018, 10, 6),
        gate_contract_ref=".planning/gate_contract_v2.3.json",
        model_candidates=candidates,
        phase_chain_audit_sha=canonical_sha,
    )
    dumped = out.model_dump()
    assert dumped["gate_contract_ref"] == ".planning/gate_contract_v2.3.json"
    assert dumped["model_candidates"] == candidates
    assert dumped["phase_chain_audit_sha"] == canonical_sha


def test_v110_additive_only():
    """Diff between live Pydantic emission and v1.0.0 file = exactly the additive trio.

    Verifies:
      - No v1.0.0 field has been removed or renamed.
      - The only NEW keys are gate_contract_ref / model_candidates /
        phase_chain_audit_sha (Phase 25 lock binding).
    """
    v100_committed = json.loads(SCHEMA_V100_PATH.read_text())
    v100_props = set(v100_committed["properties"].keys())
    assert v100_props == EXPECTED_V100_PROPERTIES, (
        f"v1.0.0 schema file drifted: extras={v100_props - EXPECTED_V100_PROPERTIES} "
        f"missing={EXPECTED_V100_PROPERTIES - v100_props}"
    )

    live = PredictorOutputV1.model_json_schema()
    live_props = set(live["properties"].keys())

    added = live_props - v100_props
    removed = v100_props - live_props
    assert removed == set(), f"v1.1.0 REMOVED v1.0.0 fields: {removed}"
    assert added == EXPECTED_V110_ADDITIVE_TRIO, (
        f"v1.1.0 added fields = {added}, expected {EXPECTED_V110_ADDITIVE_TRIO}"
    )


def test_v110_preserves_extra_ignore():
    """RESEARCH Security Domain V5 — model_config must remain extra=ignore.

    Phase 32 D-04 must NOT loosen the schema (extra=allow would allow
    schema-bypass payloads). T-32-09 mitigation.
    """
    cfg = PredictorOutputV1.model_config
    extra = cfg.get("extra")
    assert extra == "ignore", (
        f"PredictorOutputV1.model_config['extra'] = {extra!r}; must remain 'ignore' "
        "(T-32-09 schema-bypass mitigation)"
    )
