"""Unit tests for rate feature computation."""

from __future__ import annotations

import pytest

from tests.features.conftest import make_fighter_round_stats, make_round_stats
from ufc_prediction.features.rates import (
    compute_rate_features,
    estimate_fight_duration_seconds,
)


class TestEstimateFightDurationSeconds:
    """Tests for fight duration estimation."""

    def test_full_three_round_fight(self) -> None:
        """3rd round finish at 4:32 = 2*300 + 272 = 872."""
        assert estimate_fight_duration_seconds(3, "4:32", 3) == 872

    def test_none_fallback(self) -> None:
        """None/None defaults to num_rounds * 300."""
        assert estimate_fight_duration_seconds(None, None, 3) == 900

    def test_first_round_finish(self) -> None:
        """1st round at 0:30 = 0*300 + 30 = 30."""
        assert estimate_fight_duration_seconds(1, "0:30", 3) == 30

    def test_five_round_fallback(self) -> None:
        """5-round fight with no finish data = 5*300 = 1500."""
        assert estimate_fight_duration_seconds(None, None, 5) == 1500

    def test_second_round_finish(self) -> None:
        """2nd round at 2:15 = 1*300 + 135 = 435."""
        assert estimate_fight_duration_seconds(2, "2:15", 3) == 435

    def test_bad_time_format_fallback(self) -> None:
        """Invalid time_finished format falls through to default."""
        assert estimate_fight_duration_seconds(2, "bad", 3) == 900


class TestComputeRateFeatures:
    """Tests for rate feature computation."""

    def test_valid_round_stats_returns_all_eight_keys(self) -> None:
        """compute_rate_features returns dict with all 8 rate keys."""
        fighter_stats = make_fighter_round_stats(1, 100, rounds=3)
        opponent_stats = make_fighter_round_stats(2, 100, rounds=3)
        result = compute_rate_features(fighter_stats, opponent_stats, 900)

        expected_keys = {
            "sig_str_per_minute",
            "total_str_per_minute",
            "td_rate",
            "td_accuracy",
            "td_defense",
            "strike_defense",
            "ctrl_time_per_fight",
            "sub_att_per_fight",
        }
        assert set(result.keys()) == expected_keys

    def test_zero_duration_returns_none_per_minute(self) -> None:
        """Zero fight_duration_seconds -> None for per-minute rates."""
        fighter_stats = make_fighter_round_stats(1, 100, rounds=3)
        opponent_stats = make_fighter_round_stats(2, 100, rounds=3)
        result = compute_rate_features(fighter_stats, opponent_stats, 0)

        assert result["sig_str_per_minute"] is None
        assert result["total_str_per_minute"] is None
        # Non per-minute features should still have values
        assert result["td_rate"] is not None
        assert result["ctrl_time_per_fight"] is not None

    def test_all_none_stats_graceful_degradation(self) -> None:
        """All-None round stat columns -> graceful None/0.0 values."""
        none_overrides = {
            "sig_strikes_landed": None,
            "sig_strikes_attempted": None,
            "head_strikes_landed": None,
            "body_strikes_landed": None,
            "leg_strikes_landed": None,
            "takedowns_landed": None,
            "takedowns_attempted": None,
            "submission_attempts": None,
            "control_time_seconds": None,
            "knockdowns": None,
            "reversals": None,
            "distance_strikes_landed": None,
            "clinch_strikes_landed": None,
            "ground_strikes_landed": None,
        }
        fighter_stats = make_fighter_round_stats(1, 100, rounds=3, overrides=none_overrides)
        opponent_stats = make_fighter_round_stats(2, 100, rounds=3, overrides=none_overrides)
        result = compute_rate_features(fighter_stats, opponent_stats, 900)

        # Should not raise, should return a dict with 8 keys
        assert len(result) == 8

    def test_td_accuracy_none_when_zero_attempted(self) -> None:
        """td_accuracy returns None when takedowns_attempted == 0."""
        fighter_stats = make_fighter_round_stats(
            1,
            100,
            rounds=3,
            overrides={"takedowns_landed": 0, "takedowns_attempted": 0},
        )
        opponent_stats = make_fighter_round_stats(2, 100, rounds=3)
        result = compute_rate_features(fighter_stats, opponent_stats, 900)

        assert result["td_accuracy"] is None

    def test_td_defense_none_when_opponent_zero_attempted(self) -> None:
        """td_defense returns None when opponent_takedowns_attempted == 0."""
        fighter_stats = make_fighter_round_stats(1, 100, rounds=3)
        opponent_stats = make_fighter_round_stats(
            2,
            100,
            rounds=3,
            overrides={"takedowns_attempted": 0, "takedowns_landed": 0},
        )
        result = compute_rate_features(fighter_stats, opponent_stats, 900)

        assert result["td_defense"] is None

    def test_strike_defense_computation(self) -> None:
        """strike_defense = 1.0 - (opp_sig_str_landed / opp_sig_str_attempted)."""
        fighter_stats = make_fighter_round_stats(1, 100, rounds=1)
        opponent_stats = make_fighter_round_stats(
            2,
            100,
            rounds=1,
            overrides={"sig_strikes_landed": 3, "sig_strikes_attempted": 10},
        )
        result = compute_rate_features(fighter_stats, opponent_stats, 300)

        assert result["strike_defense"] is not None
        assert abs(result["strike_defense"] - 0.7) < 0.001

    def test_total_str_per_minute_from_positional(self) -> None:
        """total_str_per_minute derived from head+body+leg (not a separate column)."""
        fighter_stats = make_fighter_round_stats(
            1,
            100,
            rounds=1,
            overrides={
                "head_strikes_landed": 10,
                "body_strikes_landed": 5,
                "leg_strikes_landed": 5,
            },
        )
        opponent_stats = make_fighter_round_stats(2, 100, rounds=1)
        result = compute_rate_features(fighter_stats, opponent_stats, 300)

        # 20 total strikes in 5 minutes = 4.0 per minute
        assert result["total_str_per_minute"] is not None
        assert abs(result["total_str_per_minute"] - 4.0) < 0.001

    def test_sig_str_per_minute_computation(self) -> None:
        """sig_str_per_minute = sig_strikes_landed / fight_minutes."""
        fighter_stats = make_fighter_round_stats(
            1,
            100,
            rounds=1,
            overrides={"sig_strikes_landed": 15},
        )
        opponent_stats = make_fighter_round_stats(2, 100, rounds=1)
        result = compute_rate_features(fighter_stats, opponent_stats, 300)

        # 15 sig strikes in 5 minutes = 3.0 per minute
        assert result["sig_str_per_minute"] is not None
        assert abs(result["sig_str_per_minute"] - 3.0) < 0.001
