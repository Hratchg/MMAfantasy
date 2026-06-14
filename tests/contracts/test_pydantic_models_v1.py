"""Unit tests for PredictorOutputV1 + PredictMatchupRequestV1 (Phase 25 Plan 25-01).

Covers Pydantic-model behavior BEFORE schema artifacts are emitted (Plan 25-02).
Schema-file round-trip + Hypothesis fuzz live in Plan 25-03 test files.
"""
from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from ufc_prediction.api.v1.models import (
    PredictMatchupRequestV1,
    PredictorOutputV1,
)


class TestPredictorOutputV1:
    def test_xgb_v2_only_response_validates(self):
        """Current production: meta fields None, base_prob == win_probability."""
        obj = PredictorOutputV1(
            schema_version="1.0.0",
            win_probability=0.6234,
            fighter_a="Khabib Nurmagomedov",
            fighter_b="Conor McGregor",
            event_date=date(2018, 10, 6),
            base_prob=0.6234,
            meta_prob=None,
            meta_learner_version=None,
            meta_skipped_reason="no_meta_artifact",
        )
        assert obj.win_probability == 0.6234
        assert obj.meta_prob is None
        assert obj.meta_skipped_reason == "no_meta_artifact"

    def test_meta_v22_active_response_validates(self):
        """Forward-compat lock: META active, base_prob != win_probability (D-09)."""
        obj = PredictorOutputV1(
            schema_version="1.0.0",
            win_probability=0.7234,
            fighter_a="Islam Makhachev",
            fighter_b="Charles Oliveira",
            event_date=date(2026, 8, 15),
            base_prob=0.6512,
            meta_prob=0.7234,
            meta_learner_version="v22.1",
            meta_skipped_reason=None,
        )
        assert obj.meta_learner_version == "v22.1"
        assert obj.base_prob != obj.win_probability
        assert obj.meta_prob == 0.7234

    def test_win_probability_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            PredictorOutputV1(
                schema_version="1.0.0", win_probability=1.5,
                fighter_a="A", fighter_b="B", event_date=date(2026, 1, 1),
            )
        with pytest.raises(ValidationError):
            PredictorOutputV1(
                schema_version="1.0.0", win_probability=-0.01,
                fighter_a="A", fighter_b="B", event_date=date(2026, 1, 1),
            )

    def test_schema_version_regex_enforced(self):
        with pytest.raises(ValidationError):
            PredictorOutputV1(
                schema_version="abc",
                win_probability=0.5, fighter_a="A", fighter_b="B",
                event_date=date(2026, 1, 1),
            )

    def test_extra_field_ignored_not_allowed(self):
        """Pydantic default extra='ignore'; extra='allow' is a Security anti-pattern."""
        obj = PredictorOutputV1.model_validate({
            "schema_version": "1.0.0",
            "win_probability": 0.5,
            "fighter_a": "A", "fighter_b": "B",
            "event_date": "2026-01-01",
            "rogue_field": "should_be_dropped",
        })
        assert "rogue_field" not in obj.model_dump()


class TestPredictMatchupRequestV1:
    def test_empty_fighter_rejected(self):
        with pytest.raises(ValidationError):
            PredictMatchupRequestV1(fighter_a="", fighter_b="B")

    def test_oversize_fighter_rejected(self):
        with pytest.raises(ValidationError):
            PredictMatchupRequestV1(fighter_a="X" * 201, fighter_b="B")

    def test_event_date_optional(self):
        obj = PredictMatchupRequestV1(fighter_a="A", fighter_b="B")
        assert obj.event_date is None
