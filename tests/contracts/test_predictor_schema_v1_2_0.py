"""Phase 35 CONTRACT-V24-02 — v1.2.0 schema-contract tests.

Mirrors the Phase 25 / Phase 32 test pattern: lock the additive surface,
verify forward-compat lock (v1.0.0 + v1.1.0 partners see prediction_metadata
as null), and emit the v1.2.0 schema artifact as a side effect of running
the contract suite.

Seven behaviors (per Plan 35-05 Task 1):
  1. PredictionMetadataV12 instantiates with all 6 fields.
  2. odds_source Literal rejects unknown values.
  3. odds_timestamp_iso + calibration_slice_brier accept None.
  4. PredictorOutputV1 with prediction_metadata=None (default) round-trips
     identically to current Phase 32 D-04 behavior.
  5. PredictorOutputV1 with populated prediction_metadata Pydantic-validates
     and model_dump() includes the 6 nested fields.
  6. PredictMatchupRequestV1.accept_schema_version Literal validates the
     three allowed values and rejects others; default "1.2.0".
  7. PredictorOutputV1 model_dump() → model_validate round-trip identity.

Plus a schema-artifact emission test that writes
src/ufc_prediction/contracts/predictor.schema.v1.2.0.json from the live
Pydantic model and asserts the JSON is well-formed.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from ufc_prediction.api.v1.models import (
    PredictionMetadataV12,
    PredictMatchupRequestV1,
    PredictorOutputV1,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONTRACTS_DIR = REPO_ROOT / "src" / "ufc_prediction" / "contracts"
SCHEMA_V120_PATH = CONTRACTS_DIR / "predictor.schema.v1.2.0.json"


# ─── Behavior 1: PredictionMetadataV12 instantiation ─────────────────────────


class TestPredictionMetadataV12:
    """Pydantic model surface for the additive v1.2.0 block."""

    def test_all_six_fields_populated(self):
        """Behavior 1 — all 6 fields populate without error."""
        meta = PredictionMetadataV12(
            odds_source="live",
            odds_timestamp_iso="2026-05-26T22:00:00+00:00",
            fighter_a_n_ufc_fights=8,
            fighter_b_n_ufc_fights=12,
            is_debutant_either=False,
            calibration_slice_brier=0.21,
        )
        assert meta.odds_source == "live"
        assert meta.odds_timestamp_iso == "2026-05-26T22:00:00+00:00"
        assert meta.fighter_a_n_ufc_fights == 8
        assert meta.fighter_b_n_ufc_fights == 12
        assert meta.is_debutant_either is False
        assert meta.calibration_slice_brier == 0.21

    def test_odds_source_literal_rejects_unknown(self):
        """Behavior 2 — odds_source Literal["live","cache","nan"]."""
        with pytest.raises(ValidationError):
            PredictionMetadataV12(
                odds_source="other",  # type: ignore[arg-type]
                fighter_a_n_ufc_fights=1,
                fighter_b_n_ufc_fights=1,
                is_debutant_either=False,
            )

    def test_odds_source_accepts_three_allowed_values(self):
        for value in ("live", "cache", "nan"):
            meta = PredictionMetadataV12(
                odds_source=value,  # type: ignore[arg-type]
                fighter_a_n_ufc_fights=1,
                fighter_b_n_ufc_fights=1,
                is_debutant_either=False,
            )
            assert meta.odds_source == value

    def test_optional_fields_accept_none(self):
        """Behavior 3 — odds_timestamp_iso + calibration_slice_brier nullable."""
        meta = PredictionMetadataV12(
            odds_source="nan",
            odds_timestamp_iso=None,
            fighter_a_n_ufc_fights=0,
            fighter_b_n_ufc_fights=0,
            is_debutant_either=True,
            calibration_slice_brier=None,
        )
        assert meta.odds_timestamp_iso is None
        assert meta.calibration_slice_brier is None

    def test_n_ufc_fights_rejects_negative(self):
        """ge=0 constraint on fight counts."""
        with pytest.raises(ValidationError):
            PredictionMetadataV12(
                odds_source="nan",
                fighter_a_n_ufc_fights=-1,
                fighter_b_n_ufc_fights=0,
                is_debutant_either=True,
            )

    def test_calibration_slice_brier_bounded_0_1(self):
        """ge=0.0, le=1.0 on Brier score."""
        with pytest.raises(ValidationError):
            PredictionMetadataV12(
                odds_source="nan",
                fighter_a_n_ufc_fights=0,
                fighter_b_n_ufc_fights=0,
                is_debutant_either=True,
                calibration_slice_brier=1.5,
            )
        with pytest.raises(ValidationError):
            PredictionMetadataV12(
                odds_source="nan",
                fighter_a_n_ufc_fights=0,
                fighter_b_n_ufc_fights=0,
                is_debutant_either=True,
                calibration_slice_brier=-0.1,
            )


# ─── Behavior 4-5: PredictorOutputV1 prediction_metadata additive ────────────


class TestPredictorOutputV1AdditiveBlock:
    """Forward-compat lock — v1.2.0 field defaults to None."""

    def test_default_prediction_metadata_is_none(self):
        """Behavior 4 — Phase 25/32 partners see prediction_metadata as None."""
        out = PredictorOutputV1(
            win_probability=0.5,
            fighter_a="X",
            fighter_b="Y",
            event_date=date(2026, 1, 1),
        )
        assert out.prediction_metadata is None

    def test_default_round_trip_preserves_v110_dict_shape(self):
        """Behavior 4 — model_dump with default leaves v1.1.0 surface intact."""
        out = PredictorOutputV1(
            win_probability=0.5,
            fighter_a="X",
            fighter_b="Y",
            event_date=date(2026, 1, 1),
        )
        dumped = out.model_dump()
        # v1.0.0 fields
        for k in (
            "schema_version", "win_probability", "fighter_a", "fighter_b",
            "event_date", "base_prob", "meta_prob", "meta_learner_version",
            "meta_skipped_reason",
        ):
            assert k in dumped, f"v1.0.0 field {k!r} missing from dump"
        # v1.1.0 additive trio defaulted None
        for k in ("gate_contract_ref", "model_candidates", "phase_chain_audit_sha"):
            assert k in dumped and dumped[k] is None
        # v1.2.0 additive block default None
        assert dumped["prediction_metadata"] is None

    def test_populated_prediction_metadata_dumps_six_fields(self):
        """Behavior 5 — model_dump() includes the 6 nested fields."""
        out = PredictorOutputV1(
            win_probability=0.6,
            fighter_a="Khabib Nurmagomedov",
            fighter_b="Conor McGregor",
            event_date=date(2018, 10, 6),
            prediction_metadata=PredictionMetadataV12(
                odds_source="cache",
                odds_timestamp_iso="2018-10-05T22:00:00+00:00",
                fighter_a_n_ufc_fights=10,
                fighter_b_n_ufc_fights=21,
                is_debutant_either=False,
                calibration_slice_brier=0.19,
            ),
        )
        dumped = out.model_dump()
        block = dumped["prediction_metadata"]
        assert block is not None
        assert block["odds_source"] == "cache"
        assert block["odds_timestamp_iso"] == "2018-10-05T22:00:00+00:00"
        assert block["fighter_a_n_ufc_fights"] == 10
        assert block["fighter_b_n_ufc_fights"] == 21
        assert block["is_debutant_either"] is False
        assert block["calibration_slice_brier"] == 0.19
        # exactly 6 keys in the nested block
        assert set(block.keys()) == {
            "odds_source",
            "odds_timestamp_iso",
            "fighter_a_n_ufc_fights",
            "fighter_b_n_ufc_fights",
            "is_debutant_either",
            "calibration_slice_brier",
        }


# ─── Behavior 6: PredictMatchupRequestV1.accept_schema_version ───────────────


class TestPredictMatchupRequestV1AcceptSchemaVersion:
    """Schema negotiation field on the request body."""

    def test_default_is_v120(self):
        req = PredictMatchupRequestV1(fighter_a="A", fighter_b="B")
        assert req.accept_schema_version == "1.2.0"

    @pytest.mark.parametrize("version", ["1.0.0", "1.1.0", "1.2.0", "1.3.0"])
    def test_accepts_four_versions(self, version):
        """Phase 52 API-V26-01 extended the Literal to admit v1.3.0
        (RFC 7807 error wrapper opt-in via Accept header). Phase 69
        API-V261-01 ships the v1.3.0 sibling JSON artifacts."""
        req = PredictMatchupRequestV1(
            fighter_a="A", fighter_b="B",
            accept_schema_version=version,  # type: ignore[arg-type]
        )
        assert req.accept_schema_version == version

    @pytest.mark.parametrize("version", ["0.9.0", "1.4.0", "2.0.0", "latest", ""])
    def test_rejects_unknown_versions(self, version):
        with pytest.raises(ValidationError):
            PredictMatchupRequestV1(
                fighter_a="A", fighter_b="B",
                accept_schema_version=version,  # type: ignore[arg-type]
            )


# ─── Behavior 7: round-trip identity ─────────────────────────────────────────


class TestPredictorOutputV1RoundTrip:
    """model_dump → model_validate is identity (v1.2.0 surface)."""

    def test_round_trip_with_populated_block(self):
        original = PredictorOutputV1(
            schema_version="1.2.0",
            win_probability=0.55,
            fighter_a="A",
            fighter_b="B",
            event_date=date(2026, 5, 27),
            base_prob=0.55,
            meta_prob=None,
            meta_learner_version=None,
            meta_skipped_reason="no_meta_artifact",
            prediction_metadata=PredictionMetadataV12(
                odds_source="live",
                odds_timestamp_iso="2026-05-27T10:00:00+00:00",
                fighter_a_n_ufc_fights=4,
                fighter_b_n_ufc_fights=7,
                is_debutant_either=False,
                calibration_slice_brier=0.22,
            ),
        )
        dumped = original.model_dump()
        rehydrated = PredictorOutputV1(**dumped)
        assert rehydrated.model_dump() == dumped

    def test_round_trip_with_none_block(self):
        original = PredictorOutputV1(
            schema_version="1.0.0",
            win_probability=0.5,
            fighter_a="A",
            fighter_b="B",
            event_date=date(2026, 5, 27),
        )
        dumped = original.model_dump()
        rehydrated = PredictorOutputV1(**dumped)
        assert rehydrated.model_dump() == dumped


# ─── Schema artifact emission (side-effect of running the suite) ─────────────


class TestSchemaArtifactEmission:
    """Assert predictor.schema.v1.2.0.json is well-formed + structurally correct.

    Mirrors the Phase 25 + Phase 32 D-04 sibling-emit pattern. The Phase 35
    plan calls out a contracts directory for the v1.2.0 artifact; per
    project precedent the file lives next to predictor.schema.v1.0.0.json
    and v1.1.0.json under src/ufc_prediction/contracts/.

    Phase 69 API-V261-01 lock binding: v1.2.0 is now BYTE-FROZEN — this
    test reads the committed file rather than overwriting it from the live
    Pydantic model (which has drifted slightly across Pydantic versions
    in JSON Schema serialization metadata such as `additionalProperties`).
    `scripts/emit_partner_contracts.py` enforces the freeze via assert-
    exists; this test enforces it indirectly by READING (never writing)
    the committed file.
    """

    def test_v120_schema_artifact_well_formed(self):
        """Read predictor.schema.v1.2.0.json + assert well-formed + structural."""
        assert SCHEMA_V120_PATH.exists(), (
            f"v1.2.0 schema missing: {SCHEMA_V120_PATH}. "
            "Phase 69+ does not re-emit v1.2.0; restore from a prior commit."
        )
        # Round-trip the file as JSON to confirm well-formed.
        parsed = json.loads(SCHEMA_V120_PATH.read_text(encoding="utf-8"))
        # Sanity — the additive block lives under properties.
        assert "prediction_metadata" in parsed["properties"], (
            "v1.2.0 schema missing prediction_metadata property"
        )
        # PredictionMetadataV12 nested schema landed under $defs.
        assert "$defs" in parsed and "PredictionMetadataV12" in parsed["$defs"], (
            "v1.2.0 schema missing PredictionMetadataV12 $defs entry"
        )
        # The 6 fields are present in the nested schema.
        nested = parsed["$defs"]["PredictionMetadataV12"]["properties"]
        assert set(nested.keys()) == {
            "odds_source",
            "odds_timestamp_iso",
            "fighter_a_n_ufc_fights",
            "fighter_b_n_ufc_fights",
            "is_debutant_either",
            "calibration_slice_brier",
        }

    def test_v120_schema_additive_over_v110(self):
        """v1.2.0 = v1.1.0 ∪ {prediction_metadata, disclaimer}; no v1.1.0
        field removed.

        Phase 38 HYGIENE-V24-02 extends v1.2.0 with the `disclaimer`
        additive (regulatory surface — non-null default, additive-only
        for partners using extra='ignore').
        """
        # Live emission (mutates none — uses model class).
        live = PredictorOutputV1.model_json_schema()
        v110_path = CONTRACTS_DIR / "predictor.schema.v1.1.0.json"
        v110 = json.loads(v110_path.read_text(encoding="utf-8"))
        v110_props = set(v110["properties"].keys())
        live_props = set(live["properties"].keys())
        removed = v110_props - live_props
        assert removed == set(), (
            f"v1.2.0 REMOVED v1.1.0 fields: {removed}"
        )
        added = live_props - v110_props
        assert added == {"prediction_metadata", "disclaimer"}, (
            f"v1.2.0 added fields = {added}; expected "
            f"{{prediction_metadata, disclaimer}}"
        )
        # prediction_metadata defaults to null (Phase 25/35 lock binding).
        assert live["properties"]["prediction_metadata"].get("default") is None, (
            "prediction_metadata must default to null (Phase 25 lock binding)"
        )
        # disclaimer defaults to the DISCLAIMER_200W constant (Phase 38).
        from ufc_prediction.api.disclaimer import DISCLAIMER_200W
        assert live["properties"]["disclaimer"].get("default") == DISCLAIMER_200W, (
            "disclaimer must default to DISCLAIMER_200W (Phase 38 HYGIENE-V24-02)"
        )
        # v1.1.0 required list preserved verbatim.
        assert set(live["required"]) == set(v110["required"]), (
            f"v1.2.0 required list drifted from v1.1.0: "
            f"v1.1.0={v110['required']} live={live['required']}"
        )

    def test_extra_fields_ignored_on_predictor_output(self):
        """extra='ignore' — unknown keys do not raise."""
        payload = {
            "win_probability": 0.5,
            "fighter_a": "A",
            "fighter_b": "B",
            "event_date": "2026-05-27",
            "unknown_extra_field": "should_be_silently_dropped",
        }
        obj = PredictorOutputV1.model_validate(payload)
        # Pydantic v2 silently drops extras (ConfigDict(extra="ignore")).
        assert not hasattr(obj, "unknown_extra_field")
