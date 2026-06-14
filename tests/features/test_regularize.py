"""Unit tests for Bayesian shrinkage regularization."""

from __future__ import annotations

import pytest

from ufc_prediction.features.regularize import (
    apply_shrinkage,
    apply_shrinkage_to_features,
)


class TestApplyShrinkage:
    """Tests for the apply_shrinkage function."""

    def test_at_min_fights_no_shrinkage(self) -> None:
        """fight_count == min_fights -> factor = 1.0, no shrinkage."""
        result = apply_shrinkage(
            value=10.0, league_mean=5.0, fight_count=5, min_fights=5,
        )
        assert result == 10.0

    def test_one_fight_heavy_shrinkage(self) -> None:
        """fight_count=1, min_fights=5 -> factor=0.2.
        Result = 5.0 + (10.0 - 5.0) * 0.2 = 6.0
        """
        result = apply_shrinkage(
            value=10.0, league_mean=5.0, fight_count=1, min_fights=5,
        )
        assert abs(result - 6.0) < 0.001

    def test_zero_fights_full_shrinkage(self) -> None:
        """fight_count=0 -> factor=0.0, fully shrunk to league mean."""
        result = apply_shrinkage(
            value=10.0, league_mean=5.0, fight_count=0, min_fights=5,
        )
        assert result == 5.0

    def test_above_min_fights_capped(self) -> None:
        """fight_count > min_fights -> factor capped at 1.0."""
        result = apply_shrinkage(
            value=10.0, league_mean=5.0, fight_count=10, min_fights=5,
        )
        assert result == 10.0

    def test_three_fights_partial_shrinkage(self) -> None:
        """fight_count=3, min_fights=5 -> factor=0.6.
        Result = 5.0 + (10.0 - 5.0) * 0.6 = 8.0
        """
        result = apply_shrinkage(
            value=10.0, league_mean=5.0, fight_count=3, min_fights=5,
        )
        assert abs(result - 8.0) < 0.001


class TestApplyShrinkageToFeatures:
    """Tests for applying shrinkage to a full feature dict."""

    def test_shrinks_all_numeric_features(self) -> None:
        """All numeric features in the dict should be shrunk."""
        features = {
            "sig_str_per_minute": 10.0,
            "td_rate": 4.0,
        }
        league_means = {
            "sig_str_per_minute": 5.0,
            "td_rate": 2.0,
        }
        result = apply_shrinkage_to_features(
            features, league_means, fight_count=1, min_fights=5,
        )
        # factor = 0.2
        # sig_str: 5.0 + (10.0 - 5.0) * 0.2 = 6.0
        # td_rate: 2.0 + (4.0 - 2.0) * 0.2 = 2.4
        assert abs(result["sig_str_per_minute"] - 6.0) < 0.001
        assert abs(result["td_rate"] - 2.4) < 0.001

    def test_none_values_preserved(self) -> None:
        """None values should pass through unchanged."""
        features = {
            "sig_str_per_minute": None,
            "td_rate": 4.0,
        }
        league_means = {
            "sig_str_per_minute": 5.0,
            "td_rate": 2.0,
        }
        result = apply_shrinkage_to_features(
            features, league_means, fight_count=1, min_fights=5,
        )
        assert result["sig_str_per_minute"] is None
        assert abs(result["td_rate"] - 2.4) < 0.001

    def test_missing_league_mean_preserves_value(self) -> None:
        """Features not in league_means dict should keep original value."""
        features = {
            "sig_str_per_minute": 10.0,
            "unknown_stat": 99.0,
        }
        league_means = {
            "sig_str_per_minute": 5.0,
        }
        result = apply_shrinkage_to_features(
            features, league_means, fight_count=1, min_fights=5,
        )
        assert result["unknown_stat"] == 99.0

    def test_string_values_skipped(self) -> None:
        """Non-numeric values like style_tag should pass through unchanged."""
        features = {
            "style_tag": "striker",
            "td_rate": 4.0,
        }
        league_means = {
            "td_rate": 2.0,
        }
        result = apply_shrinkage_to_features(
            features, league_means, fight_count=1, min_fights=5,
        )
        assert result["style_tag"] == "striker"
        assert abs(result["td_rate"] - 2.4) < 0.001

    def test_sufficient_fights_no_shrinkage(self) -> None:
        """fight_count >= min_fights -> values unchanged."""
        features = {
            "sig_str_per_minute": 10.0,
            "td_rate": 4.0,
        }
        league_means = {
            "sig_str_per_minute": 5.0,
            "td_rate": 2.0,
        }
        result = apply_shrinkage_to_features(
            features, league_means, fight_count=5, min_fights=5,
        )
        assert result["sig_str_per_minute"] == 10.0
        assert result["td_rate"] == 4.0
