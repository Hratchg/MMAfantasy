"""Unit tests for opponent-adjusted performance rates."""

from __future__ import annotations

import pytest

from ufc_prediction.features.opponent_adjusted import compute_opponent_adjusted


class TestComputeOpponentAdjusted:
    """Tests for compute_opponent_adjusted covering normal, zero, and None cases."""

    def test_normal_division(self) -> None:
        """10.0 / 5.0 = 2.0 (fighter performed 2x opponent average)."""
        assert compute_opponent_adjusted(10.0, 5.0) == pytest.approx(2.0)

    def test_below_average_performance(self) -> None:
        """3.0 / 6.0 = 0.5 (fighter performed below opponent average)."""
        assert compute_opponent_adjusted(3.0, 6.0) == pytest.approx(0.5)

    def test_zero_opponent_average_returns_raw(self) -> None:
        """Zero guard per D-07: if opponent_avg is 0, return actual."""
        assert compute_opponent_adjusted(5.0, 0.0) == pytest.approx(5.0)

    def test_none_opponent_average_returns_raw(self) -> None:
        """None guard per D-07: if opponent_avg is None, return actual."""
        assert compute_opponent_adjusted(5.0, None) == pytest.approx(5.0)

    def test_zero_fighter_performance(self) -> None:
        """Fighter had zero performance: 0.0 / 5.0 = 0.0."""
        assert compute_opponent_adjusted(0.0, 5.0) == pytest.approx(0.0)
