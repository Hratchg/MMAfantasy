"""Unit tests for style classification from domain Elo differential."""

from __future__ import annotations

import pytest

from ufc_prediction.features.style import classify_style


class TestClassifyStyle:
    """Tests for classify_style covering all style tags and edge cases."""

    def test_striker_when_striking_elo_exceeds_threshold(self) -> None:
        """striking - grappling = 50 > 30 threshold -> striker."""
        assert classify_style(1550.0, 1500.0, threshold=30.0) == "striker"

    def test_grappler_when_grappling_elo_exceeds_threshold(self) -> None:
        """striking - grappling = -50 < -30 threshold -> grappler."""
        assert classify_style(1500.0, 1550.0, threshold=30.0) == "grappler"

    def test_balanced_when_within_threshold(self) -> None:
        """striking - grappling = 10, within 30 threshold -> balanced."""
        assert classify_style(1510.0, 1500.0, threshold=30.0) == "balanced"

    def test_balanced_when_striking_elo_is_none(self) -> None:
        """Missing striking Elo -> balanced."""
        assert classify_style(None, 1500.0) == "balanced"

    def test_balanced_when_grappling_elo_is_none(self) -> None:
        """Missing grappling Elo -> balanced."""
        assert classify_style(1500.0, None) == "balanced"

    def test_balanced_when_both_elo_are_none(self) -> None:
        """Both Elo values missing -> balanced."""
        assert classify_style(None, None) == "balanced"

    def test_balanced_at_exact_threshold_boundary(self) -> None:
        """diff = 30 exactly, not strictly > 30, -> balanced."""
        assert classify_style(1530.0, 1500.0, threshold=30.0) == "balanced"

    def test_striker_just_above_threshold(self) -> None:
        """diff = 31 > 30 -> striker."""
        assert classify_style(1531.0, 1500.0, threshold=30.0) == "striker"

    def test_grappler_at_exact_negative_threshold_boundary(self) -> None:
        """diff = -30 exactly, not strictly < -30, -> balanced."""
        assert classify_style(1500.0, 1530.0, threshold=30.0) == "balanced"

    def test_grappler_just_below_negative_threshold(self) -> None:
        """diff = -31 < -30 -> grappler."""
        assert classify_style(1500.0, 1531.0, threshold=30.0) == "grappler"
