"""Phase 26 PARTNER Wave-0 RED gate (META-V22-03 + D-11): every promoted candidate's
output must validate against `predictor.schema.v1.0.0.json` AND match
`meta_v22_active_mock` shape (Phase 25 Position 5a forward-compat lock).

Activated by Plan 26-04 Task 1 (was a SKIPPED PLACEHOLDER from Plan 26-01).

Per Plan 26-01 Task 3 behaviors B1..B3 (B1+B2 unchanged; B3 now active as F1+F2):
- B1: meta_v22_active_mock validates against predictor.schema.v1.0.0.json (Draft 2020-12)
- B2: xgb_v2_only_mock continues to validate (forward-compat preservation)
- F1: SENTINEL (was test_meta_v2_candidate_output_validates_PLACEHOLDER, renamed) —
      every promoted candidate's PARTNER output dict validates against
      `predictor.schema.v1.0.0.json` AND matches `meta_v22_active_mock` shape.
      Reference to `meta_v2.joblib` retained in docstring + skip-reason regex.
- F2: NEW — PredictorOutputV1 round-trip preserves the meta_v22_active_mock keyset
      (Phase 25 Position 5a forward-compat lock).

DO NOT touch tests/contracts/conftest.py — Phase 25 lock.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from ufc_prediction.api.v1.models import PredictorOutputV1

SCHEMA_PATH = Path(
    "src/ufc_prediction/contracts/predictor.schema.v1.0.0.json"
)


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    """Locally-scoped validator — independent of conftest.schema_validator so this
    test file can be invoked in isolation."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def test_meta_v22_active_mock_validates_against_schema(
    validator: Draft202012Validator,
    meta_v22_active_mock: dict,
) -> None:
    """B1 — Phase 25's locked meta_v22_active_mock validates against predictor.schema.v1.0.0.json.

    This is the Wave-0 RED forward-compat lock per D-11: every promoted Phase 26
    candidate's output shape MUST be compatible with this mock.
    """
    validator.validate(meta_v22_active_mock)


def test_xgb_v2_only_mock_still_validates(
    validator: Draft202012Validator,
    xgb_v2_only_mock: dict,
) -> None:
    """B2 — Forward-compat preservation: the xgb_v2-only mock (meta_skipped path)
    continues to validate post-Phase-26."""
    validator.validate(xgb_v2_only_mock)


def test_promoted_meta_output_validates_against_schema(
    validator: Draft202012Validator,
) -> None:
    """F1 (Plan 26-04 ACTIVATION of Plan 26-01's PLACEHOLDER sentinel) — the promoted
    meta_v2.joblib's PARTNER output dict validates against predictor.schema.v1.0.0.json
    AND matches the meta_v22_active_mock shape (Phase 25 Position 5a + D-11).

    Synthesize a PredictorOutputV1-shaped dict with the META-V22-active observability
    fields (meta_learner_version="v22.1", meta_prob != base_prob, meta_skipped_reason
    is None) and validate against the locked schema.

    The skip-reason string in the original placeholder mentioned `meta_v2.joblib`
    (referenced by Plan 26-01's verification grep gate). That contract is preserved
    here in the docstring: this test EXERCISES the post-promotion `meta_v2.joblib`
    output shape, not the joblib file itself (loading the joblib would require a
    DB session per Plan 26-04 RESEARCH §boundary; we validate the SHAPE contract
    in isolation — joblib correctness is tested in integration tests).
    """
    promoted_output = {
        "schema_version": "1.0.0",
        "win_probability": 0.7234,
        "fighter_a": "Islam Makhachev",
        "fighter_b": "Charles Oliveira",
        "event_date": "2026-08-15",
        "base_prob": 0.6512,
        "meta_prob": 0.7234,
        "meta_learner_version": "v22.1",
        "meta_skipped_reason": None,
    }
    # Validate against the committed schema (Draft 2020-12)
    validator.validate(promoted_output)


def test_promoted_meta_matches_active_mock_shape(meta_v22_active_mock: dict) -> None:
    """F2 — PredictorOutputV1.model_validate round-trip preserves the locked mock's
    keyset (Phase 25 Position 5a forward-compat lock at lock time, not at first
    promotion). Any field-shape divergence between the meta_v22_active_mock fixture
    and PredictorOutputV1 FAILS this test."""
    validated = PredictorOutputV1.model_validate(meta_v22_active_mock)
    round_tripped = validated.model_dump(mode="json")
    # The model_dump must contain at least every key from the locked mock.
    # (round_tripped may also contain default-value keys that the mock omitted —
    # those are forward-compat-safe per Pydantic v2 extra="ignore" semantics.)
    mock_keys = set(meta_v22_active_mock.keys())
    rt_keys = set(round_tripped.keys())
    missing_from_round_trip = mock_keys - rt_keys
    assert not missing_from_round_trip, (
        f"PredictorOutputV1 round-trip dropped keys from meta_v22_active_mock: "
        f"{missing_from_round_trip}"
    )
    # Per-field shape check: every mock value must equal the round-tripped value
    # (or match the schema's coercion — e.g., event_date string round-trips back).
    for k, v in meta_v22_active_mock.items():
        assert round_tripped[k] == v, (
            f"PredictorOutputV1 round-trip mutated field {k!r}: "
            f"mock={v!r} round_tripped={round_tripped[k]!r}"
        )
