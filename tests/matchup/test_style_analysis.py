"""Unit tests for style index and style-vs-style win rates."""

from __future__ import annotations

from ufc_prediction.matchup.style_analysis import (
    StyleMatchupWinRate,
    compute_style_index,
    compute_style_vs_style_win_rates,
)


class TestStyleIndex:
    """Tests for compute_style_index."""

    def test_striker_positive_index(self):
        """striking_elo=1550, grappling_elo=1480 -> index = 70.0."""
        result = compute_style_index(1550.0, 1480.0)
        assert result == 70.0

    def test_grappler_negative_index(self):
        """striking_elo=1450, grappling_elo=1530 -> index = -80.0."""
        result = compute_style_index(1450.0, 1530.0)
        assert result == -80.0

    def test_striking_elo_none(self):
        """striking_elo is None -> index is None."""
        result = compute_style_index(None, 1500.0)
        assert result is None

    def test_grappling_elo_none(self):
        """grappling_elo is None -> index is None."""
        result = compute_style_index(1500.0, None)
        assert result is None

    def test_both_elo_none(self):
        """Both Elo None -> index is None."""
        result = compute_style_index(None, None)
        assert result is None


class TestStyleVsStyleWinRates:
    """Tests for compute_style_vs_style_win_rates."""

    def test_aggregation_with_sufficient_data(self):
        """Aggregation with sample counts and win percentages."""
        matchup_counts = {
            ("grappler", "striker"): {"a_wins": 6, "b_wins": 14, "total": 20},
            ("balanced", "striker"): {"a_wins": 8, "b_wins": 12, "total": 20},
        }

        result = compute_style_vs_style_win_rates(matchup_counts)

        assert isinstance(result, list)
        assert all(isinstance(r, StyleMatchupWinRate) for r in result)

        # Find grappler vs striker
        gs = next(r for r in result if r.style_a == "grappler" and r.style_b == "striker")
        assert gs.style_a_wins == 6
        assert gs.style_b_wins == 14
        assert gs.total == 20
        assert gs.style_a_win_pct is not None
        assert abs(gs.style_a_win_pct - 0.30) < 0.01
        assert gs.sufficient_data is True

    def test_below_threshold_insufficient_data(self):
        """Below 10-fight threshold -> insufficient_data flag per D-09."""
        matchup_counts = {
            ("grappler", "striker"): {"a_wins": 3, "b_wins": 4, "total": 7},
        }

        result = compute_style_vs_style_win_rates(matchup_counts)

        gs = next(r for r in result if r.style_a == "grappler" and r.style_b == "striker")
        assert gs.style_a_win_pct is None
        assert gs.sufficient_data is False
        assert gs.total == 7

    def test_mixed_matchup_types(self):
        """Mixed style matchup types counted correctly."""
        matchup_counts = {
            ("balanced", "grappler"): {"a_wins": 5, "b_wins": 5, "total": 10},
            ("balanced", "striker"): {"a_wins": 3, "b_wins": 2, "total": 5},
            ("grappler", "striker"): {"a_wins": 12, "b_wins": 8, "total": 20},
        }

        result = compute_style_vs_style_win_rates(matchup_counts)

        # Should have 3 entries
        assert len(result) == 3

        # sorted alphabetically by (style_a, style_b)
        assert result[0].style_a == "balanced"
        assert result[0].style_b == "grappler"
        assert result[1].style_a == "balanced"
        assert result[1].style_b == "striker"
        assert result[2].style_a == "grappler"
        assert result[2].style_b == "striker"

        # balanced vs grappler: exactly 10, sufficient
        assert result[0].sufficient_data is True
        assert result[0].style_a_win_pct is not None
        assert abs(result[0].style_a_win_pct - 0.50) < 0.01

        # balanced vs striker: 5 < 10, insufficient
        assert result[1].sufficient_data is False
        assert result[1].style_a_win_pct is None

        # grappler vs striker: 20, sufficient
        assert result[2].sufficient_data is True
        assert abs(result[2].style_a_win_pct - 0.60) < 0.01

    def test_empty_matchup_counts(self):
        """Empty matchup counts -> empty result list."""
        result = compute_style_vs_style_win_rates({})
        assert result == []
