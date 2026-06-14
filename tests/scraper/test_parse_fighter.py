"""Tests for fighter profile page parser."""

from __future__ import annotations

import pytest

from ufc_prediction.scraper.parse_fighter import parse_fighter_profile
from ufc_prediction.scraper.models import FighterProfile


class TestParseFighterProfile:
    """Tests using the complete fighter profile fixture."""

    def test_parse_fighter_name(self, fighter_profile_html: str) -> None:
        """Name is Carlos Ulberg."""
        profile = parse_fighter_profile(fighter_profile_html)
        assert isinstance(profile, FighterProfile)
        assert profile.name == "Carlos Ulberg"

    def test_parse_fighter_nickname(self, fighter_profile_html: str) -> None:
        """Nickname is Black Jag."""
        profile = parse_fighter_profile(fighter_profile_html)
        assert profile.nickname == "Black Jag"

    def test_parse_fighter_height(self, fighter_profile_html: str) -> None:
        """height_str contains 6' 4."""
        profile = parse_fighter_profile(fighter_profile_html)
        assert profile.height_str is not None
        assert "6' 4" in profile.height_str

    def test_parse_fighter_weight(self, fighter_profile_html: str) -> None:
        """weight_str contains 205."""
        profile = parse_fighter_profile(fighter_profile_html)
        assert profile.weight_str is not None
        assert "205" in profile.weight_str

    def test_parse_fighter_reach(self, fighter_profile_html: str) -> None:
        """reach_str contains 77."""
        profile = parse_fighter_profile(fighter_profile_html)
        assert profile.reach_str is not None
        assert "77" in profile.reach_str

    def test_parse_fighter_stance(self, fighter_profile_html: str) -> None:
        """Stance is Orthodox."""
        profile = parse_fighter_profile(fighter_profile_html)
        assert profile.stance == "Orthodox"

    def test_parse_fighter_dob(self, fighter_profile_html: str) -> None:
        """dob_str contains Nov 07, 1990."""
        profile = parse_fighter_profile(fighter_profile_html)
        assert profile.dob_str is not None
        assert "Nov 07, 1990" in profile.dob_str


class TestParseFighterProfileMissing:
    """Tests using the fighter profile with missing fields fixture."""

    def test_parse_fighter_missing_height(self, fighter_profile_missing_html: str) -> None:
        """Missing profile has height_str as '--'."""
        profile = parse_fighter_profile(fighter_profile_missing_html)
        assert profile.height_str == "--"

    def test_parse_fighter_missing_stance(self, fighter_profile_missing_html: str) -> None:
        """Missing profile has stance as None."""
        profile = parse_fighter_profile(fighter_profile_missing_html)
        assert profile.stance is None

    def test_parse_fighter_missing_dob(self, fighter_profile_missing_html: str) -> None:
        """Missing profile has dob_str as None."""
        profile = parse_fighter_profile(fighter_profile_missing_html)
        assert profile.dob_str is None

    def test_parse_fighter_missing_nickname(self, fighter_profile_missing_html: str) -> None:
        """Missing profile has nickname as None."""
        profile = parse_fighter_profile(fighter_profile_missing_html)
        assert profile.nickname is None

    def test_parse_fighter_missing_reach(self, fighter_profile_missing_html: str) -> None:
        """Missing profile has reach_str as '--'."""
        profile = parse_fighter_profile(fighter_profile_missing_html)
        assert profile.reach_str == "--"
