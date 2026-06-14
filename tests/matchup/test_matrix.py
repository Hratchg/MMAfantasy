"""Unit tests for striking and grappling matrix computation."""

from __future__ import annotations

from ufc_prediction.matchup.matrix import (
    GrapplingMatrix,
    StrikingMatrix,
    compute_grappling_matrix,
    compute_striking_matrix,
)


class TestStrikingMatrix:
    """Tests for compute_striking_matrix."""

    def test_both_fighters_ewma_features(self, sample_features_full):
        """Both fighters have EWMA features -> uses EWMA values."""
        features_a = sample_features_full.copy()
        features_b = sample_features_full.copy()
        features_b["sig_str_per_minute_ewma"] = 3.8
        features_b["strike_defense_ewma"] = 0.55

        result = compute_striking_matrix(
            features_a, features_b, accuracy_a=0.48, accuracy_b=0.42
        )

        assert isinstance(result, StrikingMatrix)
        # A's offense uses EWMA sig_str
        assert result.a_sig_str_per_min == 4.5
        assert result.a_striking_accuracy == 0.48
        # A's defense uses EWMA strike_defense
        assert result.a_strike_defense == 0.60
        # B's offense uses EWMA sig_str
        assert result.b_sig_str_per_min == 3.8
        assert result.b_striking_accuracy == 0.42
        assert result.b_strike_defense == 0.55

    def test_fighter_a_missing_ewma_falls_back(self, sample_features_partial):
        """Fighter A missing EWMA -> falls back to career average keys (D-05)."""
        features_a = sample_features_partial.copy()
        features_b = {
            "sig_str_per_minute_ewma": 5.0,
            "strike_defense_ewma": 0.70,
        }

        result = compute_striking_matrix(
            features_a, features_b, accuracy_a=0.45, accuracy_b=0.50
        )

        # A falls back to career average
        assert result.a_sig_str_per_min == 4.0  # career avg
        assert result.a_strike_defense == 0.55  # career avg
        # B uses EWMA
        assert result.b_sig_str_per_min == 5.0
        assert result.b_strike_defense == 0.70

    def test_fighter_no_features(self):
        """Fighter with no features at all -> all values None."""
        result = compute_striking_matrix(None, None)

        assert result.a_sig_str_per_min is None
        assert result.a_striking_accuracy is None
        assert result.a_strike_defense is None
        assert result.b_sig_str_per_min is None
        assert result.b_striking_accuracy is None
        assert result.b_strike_defense is None

    def test_matrix_structure(self, sample_features_full):
        """Matrix has a_offense vs b_defense and b_offense vs a_defense."""
        features_a = sample_features_full.copy()
        features_b = sample_features_full.copy()

        result = compute_striking_matrix(
            features_a, features_b, accuracy_a=0.50, accuracy_b=0.45
        )

        # Verify all fields exist
        assert hasattr(result, "a_sig_str_per_min")
        assert hasattr(result, "a_striking_accuracy")
        assert hasattr(result, "a_strike_defense")
        assert hasattr(result, "b_sig_str_per_min")
        assert hasattr(result, "b_striking_accuracy")
        assert hasattr(result, "b_strike_defense")


class TestGrapplingMatrix:
    """Tests for compute_grappling_matrix."""

    def test_both_fighters_ewma_features(self, sample_features_full):
        """Both fighters have EWMA features -> uses EWMA values."""
        features_a = sample_features_full.copy()
        features_b = sample_features_full.copy()
        features_b["td_rate_ewma"] = 1.8
        features_b["td_accuracy_ewma"] = 0.40
        features_b["td_defense_ewma"] = 0.60
        features_b["ctrl_time_per_fight_ewma"] = 100.0

        result = compute_grappling_matrix(features_a, features_b)

        assert isinstance(result, GrapplingMatrix)
        # A's offense uses EWMA
        assert result.a_td_rate == 2.3
        assert result.a_td_accuracy == 0.50
        assert result.a_td_defense == 0.70
        assert result.a_ctrl_time == 130.0
        # B's uses EWMA
        assert result.b_td_rate == 1.8
        assert result.b_td_accuracy == 0.40
        assert result.b_td_defense == 0.60
        assert result.b_ctrl_time == 100.0

    def test_fallback_to_career_averages(self, sample_features_partial):
        """Fallback to career averages when EWMA missing (per D-07)."""
        features_a = sample_features_partial.copy()
        features_b = sample_features_partial.copy()

        result = compute_grappling_matrix(features_a, features_b)

        # Falls back to career averages
        assert result.a_td_rate == 2.0
        assert result.a_td_accuracy == 0.45
        assert result.a_td_defense == 0.65
        assert result.a_ctrl_time == 120.0
        assert result.b_td_rate == 2.0
        assert result.b_td_accuracy == 0.45

    def test_matrix_structure(self, sample_features_full):
        """Matrix has a_offense vs b_defense and b_offense vs a_defense, plus ctrl_time."""
        features_a = sample_features_full.copy()
        features_b = sample_features_full.copy()

        result = compute_grappling_matrix(features_a, features_b)

        # All fields present
        assert hasattr(result, "a_td_rate")
        assert hasattr(result, "a_td_accuracy")
        assert hasattr(result, "a_td_defense")
        assert hasattr(result, "b_td_rate")
        assert hasattr(result, "b_td_accuracy")
        assert hasattr(result, "b_td_defense")
        assert hasattr(result, "a_ctrl_time")
        assert hasattr(result, "b_ctrl_time")

    def test_none_features(self):
        """None features -> all values None."""
        result = compute_grappling_matrix(None, None)

        assert result.a_td_rate is None
        assert result.a_td_accuracy is None
        assert result.a_td_defense is None
        assert result.a_ctrl_time is None
        assert result.b_td_rate is None
        assert result.b_td_accuracy is None
        assert result.b_td_defense is None
        assert result.b_ctrl_time is None
