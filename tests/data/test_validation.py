"""Tests for Pydantic validation schemas (DATA-05).

Validates that FightRow, FightStats, FighterRow, and IngestResult behave
correctly: accepting valid data, rejecting invalid data, and tracking
ingestion statistics.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from ufc_prediction.data.schemas import (
    VALID_WEIGHT_CLASSES,
    FighterRow,
    FightRow,
    FightStats,
    IngestResult,
)

# ── FightRow validation ──────────────────────────────────────────────────────


def _valid_fight_kwargs() -> dict:
    """Return a valid FightRow kwargs dict for test reuse."""
    return {
        "event_name": "UFC 300",
        "event_date": date(2024, 4, 13),
        "location": "Las Vegas, Nevada, USA",
        "fighter_a_name": "Jon Jones",
        "fighter_b_name": "Stipe Miocic",
        "winner_name": "Jon Jones",
        "weight_class": "Heavyweight",
        "method": "KO/TKO",
        "method_detail": "Spinning back kick",
        "round_finished": 3,
        "time_finished": "4:25",
        "is_title_fight": True,
        "num_rounds": 5,
    }


def test_fight_row_valid():
    """FightRow accepts valid fight data with all fields populated."""
    row = FightRow(**_valid_fight_kwargs())
    assert row.event_name == "UFC 300"
    assert row.weight_class == "Heavyweight"
    assert row.is_title_fight is True
    assert row.fighter_a_name == "Jon Jones"
    assert row.fighter_b_name == "Stipe Miocic"


def test_fight_row_rejects_invalid_weight_class():
    """FightRow rejects unknown weight class not in VALID_WEIGHT_CLASSES."""
    kwargs = _valid_fight_kwargs()
    kwargs["weight_class"] = "SuperDuperWeight"
    with pytest.raises(ValidationError, match="Unknown weight class"):
        FightRow(**kwargs)


def test_fight_row_rejects_future_date():
    """FightRow rejects event_date in the future per D-07 temporal check."""
    kwargs = _valid_fight_kwargs()
    kwargs["event_date"] = date.today() + timedelta(days=30)
    with pytest.raises(ValidationError, match="in the future"):
        FightRow(**kwargs)


def test_fight_row_accepts_today():
    """FightRow accepts event_date of today (not in the future)."""
    kwargs = _valid_fight_kwargs()
    kwargs["event_date"] = date.today()
    row = FightRow(**kwargs)
    assert row.event_date == date.today()


def test_fight_row_accepts_tomorrow_timezone_slack():
    """FightRow accepts tomorrow's date (1-day timezone slack)."""
    kwargs = _valid_fight_kwargs()
    kwargs["event_date"] = date.today() + timedelta(days=1)
    row = FightRow(**kwargs)
    assert row.event_date == date.today() + timedelta(days=1)


def test_fight_row_draw_winner_none():
    """FightRow accepts winner_name=None for draws/NC."""
    kwargs = _valid_fight_kwargs()
    kwargs["winner_name"] = None
    row = FightRow(**kwargs)
    assert row.winner_name is None


def test_fight_row_all_weight_classes():
    """FightRow accepts all 14 valid weight classes."""
    for wc in VALID_WEIGHT_CLASSES:
        kwargs = _valid_fight_kwargs()
        kwargs["weight_class"] = wc
        row = FightRow(**kwargs)
        assert row.weight_class == wc


def test_fight_row_with_stats():
    """FightRow accepts embedded FightStats for both corners."""
    kwargs = _valid_fight_kwargs()
    kwargs["fighter_a_stats"] = FightStats(
        sig_strikes_landed=69,
        sig_strikes_attempted=101,
        takedowns_landed=2,
        takedowns_attempted=5,
    )
    kwargs["fighter_b_stats"] = FightStats(
        sig_strikes_landed=48,
        sig_strikes_attempted=89,
    )
    row = FightRow(**kwargs)
    assert row.fighter_a_stats.sig_strikes_landed == 69
    assert row.fighter_b_stats.sig_strikes_landed == 48


# ── FighterRow validation ────────────────────────────────────────────────────


def test_fighter_row_valid():
    """FighterRow accepts valid fighter with all profile fields."""
    row = FighterRow(
        name="Jon Jones",
        height_inches=76.0,
        reach_inches=84.5,
        stance="Orthodox",
        date_of_birth=date(1987, 7, 19),
    )
    assert row.name == "Jon Jones"
    assert row.height_inches == 76.0
    assert row.reach_inches == 84.5
    assert row.stance == "Orthodox"
    assert row.date_of_birth == date(1987, 7, 19)


def test_fighter_row_all_optional_none():
    """FighterRow accepts fighter with all None optional fields per D-04 nullable."""
    row = FighterRow(
        name="Unknown Fighter",
        height_inches=None,
        reach_inches=None,
        leg_reach_inches=None,
        stance=None,
        date_of_birth=None,
    )
    assert row.name == "Unknown Fighter"
    assert row.height_inches is None
    assert row.reach_inches is None
    assert row.stance is None
    assert row.date_of_birth is None


# ── FightStats validation ────────────────────────────────────────────────────


def test_fight_stats_all_none():
    """FightStats accepts all defaults (None) per D-04 nullable."""
    stats = FightStats()
    assert stats.sig_strikes_landed is None
    assert stats.sig_strikes_attempted is None
    assert stats.takedowns_landed is None
    assert stats.takedowns_attempted is None
    assert stats.submission_attempts is None
    assert stats.reversals is None
    assert stats.control_time_seconds is None
    assert stats.knockdowns is None
    assert stats.head_strikes_landed is None
    assert stats.body_strikes_landed is None
    assert stats.leg_strikes_landed is None
    assert stats.distance_strikes_landed is None
    assert stats.clinch_strikes_landed is None
    assert stats.ground_strikes_landed is None


def test_fight_stats_populated():
    """FightStats accepts fully populated stat fields."""
    stats = FightStats(
        sig_strikes_landed=69,
        sig_strikes_attempted=101,
        takedowns_landed=2,
        takedowns_attempted=5,
        submission_attempts=1,
        reversals=0,
        control_time_seconds=195,
        knockdowns=2,
        head_strikes_landed=40,
        body_strikes_landed=10,
        leg_strikes_landed=19,
        distance_strikes_landed=50,
        clinch_strikes_landed=10,
        ground_strikes_landed=9,
    )
    assert stats.sig_strikes_landed == 69
    assert stats.control_time_seconds == 195


# ── IngestResult tracking ────────────────────────────────────────────────────


def test_ingest_result_tracking():
    """IngestResult correctly tracks accepted, rejected, updated counts."""
    result = IngestResult()
    assert result.accepted == 0
    assert result.rejected == 0
    assert result.updated == 0

    result.accepted += 5
    result.updated += 2
    assert result.accepted == 5
    assert result.updated == 2


def test_ingest_result_log_rejection():
    """IngestResult.log_rejection records row index and error string."""
    result = IngestResult()
    result.log_rejection(
        row_index=42,
        raw_row={"R_fighter": "Jon Jones", "B_fighter": "Stipe Miocic", "date": "2024-04-13"},
        error="Validation error: unknown weight class",
    )
    assert result.rejected == 1
    assert len(result.rejections) == 1
    assert result.rejections[0]["row"] == 42
    assert "Validation error" in result.rejections[0]["error"]
    assert "Jon Jones" in result.rejections[0]["data"]["R_fighter"]


def test_ingest_result_log_rejection_truncates_data():
    """IngestResult.log_rejection truncates long data values to 100 chars."""
    result = IngestResult()
    result.log_rejection(
        row_index=0,
        raw_row={"R_fighter": "A" * 200, "B_fighter": "B" * 200, "date": "2024-01-01"},
        error="x" * 600,
    )
    assert len(result.rejections[0]["data"]["R_fighter"]) <= 100
    assert len(result.rejections[0]["error"]) <= 500


def test_ingest_result_summary_format():
    """IngestResult.summary() returns formatted string with counts."""
    result = IngestResult()
    result.accepted = 100
    result.updated = 15
    result.rejected = 3
    summary = result.summary()
    assert "100 accepted" in summary
    assert "15 updated" in summary
    assert "3 rejected" in summary
