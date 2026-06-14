"""Unit tests for matchup orchestrator (build_matchup) and MatchupResult."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ufc_prediction.matchup.compare import (
    EloComparison,
    MatchupResult,
    StyleAnalysis,
    build_matchup,
)
from ufc_prediction.matchup.matrix import GrapplingMatrix, StrikingMatrix
from ufc_prediction.matchup.physical import PhysicalDifferentials


@pytest.fixture()
def fighter_a():
    """Fighter A with full physical data."""
    return SimpleNamespace(
        id=1,
        name="Fighter A",
        height_inches=73.0,
        reach_inches=75.0,
        leg_reach_inches=43.0,
        date_of_birth=date(1990, 1, 1),
    )


@pytest.fixture()
def fighter_b():
    """Fighter B with full physical data."""
    return SimpleNamespace(
        id=2,
        name="Fighter B",
        height_inches=70.0,
        reach_inches=72.0,
        leg_reach_inches=40.0,
        date_of_birth=date(1988, 6, 15),
    )


@pytest.fixture()
def full_features():
    """Complete features JSON."""
    return {
        "sig_str_per_minute": 4.0,
        "sig_str_per_minute_ewma": 4.5,
        "strike_defense": 0.55,
        "strike_defense_ewma": 0.60,
        "td_rate": 2.0,
        "td_rate_ewma": 2.3,
        "td_accuracy": 0.45,
        "td_accuracy_ewma": 0.50,
        "td_defense": 0.65,
        "td_defense_ewma": 0.70,
        "ctrl_time_per_fight": 120.0,
        "ctrl_time_per_fight_ewma": 130.0,
        "style_tag": "striker",
    }


def _make_mock_session():
    """Create a mock session object (not used directly by build_matchup)."""
    return SimpleNamespace()


class TestBuildMatchupFullyFeatured:
    """Test build_matchup with two fully-featured fighters."""

    @patch("ufc_prediction.matchup.compare.get_style_matchup_counts")
    @patch("ufc_prediction.matchup.compare.get_striking_accuracy")
    @patch("ufc_prediction.matchup.compare.get_latest_features")
    @patch("ufc_prediction.matchup.compare.get_fighter_domain_elo")
    @patch("ufc_prediction.matchup.compare.get_fighter_detail")
    @patch("ufc_prediction.matchup.compare.get_fighter_divisions")
    def test_all_sections_populated(
        self,
        mock_divisions,
        mock_detail,
        mock_domain_elo,
        mock_features,
        mock_accuracy,
        mock_style_counts,
        fighter_a,
        fighter_b,
        full_features,
    ):
        """build_matchup with two fully-featured fighters -> all sections populated."""
        mock_divisions.return_value = ["Lightweight"]
        mock_detail.return_value = {
            "elo": 1550.0,
            "division": "Lightweight",
            "wins": 10,
            "losses": 2,
            "total_fights": 12,
            "last_fight_date": date(2024, 6, 1),
        }
        mock_domain_elo.return_value = {
            "striking_elo": 1550.0,
            "grappling_elo": 1480.0,
        }
        mock_features.return_value = full_features
        mock_accuracy.return_value = 0.48
        mock_style_counts.return_value = {
            ("grappler", "striker"): {"a_wins": 6, "b_wins": 14, "total": 20},
        }

        session = _make_mock_session()
        result = build_matchup(session, fighter_a, fighter_b)

        assert isinstance(result, MatchupResult)
        assert result.fighter_a_name == "Fighter A"
        assert result.fighter_b_name == "Fighter B"

        # Physical section populated
        assert isinstance(result.physical, PhysicalDifferentials)
        assert result.physical.height_diff == 3.0
        assert result.physical.reach_diff == 3.0

        # Elo section populated
        assert isinstance(result.elo, EloComparison)
        assert result.elo.fighter_a_overall == 1550.0
        assert result.elo.fighter_b_overall == 1550.0
        assert result.elo.fighter_a_striking == 1550.0
        assert result.elo.fighter_a_grappling == 1480.0

        # Striking section populated
        assert isinstance(result.striking, StrikingMatrix)
        assert result.striking.a_sig_str_per_min == 4.5

        # Grappling section populated
        assert isinstance(result.grappling, GrapplingMatrix)
        assert result.grappling.a_td_rate == 2.3

        # Style section populated
        assert isinstance(result.style, StyleAnalysis)
        assert result.style.fighter_a_style == "striker"
        assert result.style.fighter_a_index == 70.0  # 1550 - 1480
        assert result.style.cross_division_warning is None


class TestBuildMatchupMissingFeatures:
    """Test build_matchup with missing features data."""

    @patch("ufc_prediction.matchup.compare.get_style_matchup_counts")
    @patch("ufc_prediction.matchup.compare.get_striking_accuracy")
    @patch("ufc_prediction.matchup.compare.get_latest_features")
    @patch("ufc_prediction.matchup.compare.get_fighter_domain_elo")
    @patch("ufc_prediction.matchup.compare.get_fighter_detail")
    @patch("ufc_prediction.matchup.compare.get_fighter_divisions")
    def test_missing_features_graceful(
        self,
        mock_divisions,
        mock_detail,
        mock_domain_elo,
        mock_features,
        mock_accuracy,
        mock_style_counts,
        fighter_a,
        fighter_b,
    ):
        """build_matchup with fighter missing features -> None values gracefully."""
        mock_divisions.return_value = ["Lightweight"]
        mock_detail.return_value = {
            "elo": None,
            "division": "Lightweight",
            "wins": 0,
            "losses": 0,
            "total_fights": 0,
            "last_fight_date": None,
        }
        mock_domain_elo.return_value = {
            "striking_elo": None,
            "grappling_elo": None,
        }
        mock_features.return_value = None
        mock_accuracy.return_value = None
        mock_style_counts.return_value = {}

        session = _make_mock_session()
        result = build_matchup(session, fighter_a, fighter_b)

        assert isinstance(result, MatchupResult)
        # Elo should be None
        assert result.elo.fighter_a_overall is None
        assert result.elo.fighter_b_overall is None
        # Striking matrix should have None values
        assert result.striking.a_sig_str_per_min is None
        assert result.striking.a_striking_accuracy is None
        # Style should have None
        assert result.style.fighter_a_style is None
        assert result.style.fighter_a_index is None


class TestBuildMatchupCrossDivision:
    """Test build_matchup with fighters in different divisions."""

    @patch("ufc_prediction.matchup.compare.get_style_matchup_counts")
    @patch("ufc_prediction.matchup.compare.get_striking_accuracy")
    @patch("ufc_prediction.matchup.compare.get_latest_features")
    @patch("ufc_prediction.matchup.compare.get_fighter_domain_elo")
    @patch("ufc_prediction.matchup.compare.get_fighter_detail")
    @patch("ufc_prediction.matchup.compare.get_fighter_divisions")
    def test_cross_division_warning(
        self,
        mock_divisions,
        mock_detail,
        mock_domain_elo,
        mock_features,
        mock_accuracy,
        mock_style_counts,
        fighter_a,
        fighter_b,
    ):
        """build_matchup with fighters in different divisions -> warning included."""
        # Fighter A in Lightweight, Fighter B in Welterweight
        mock_divisions.side_effect = lambda session, fighter_id: (
            ["Lightweight"] if fighter_id == 1 else ["Welterweight"]
        )
        mock_detail.return_value = {
            "elo": 1500.0,
            "division": "Lightweight",
            "wins": 5,
            "losses": 3,
            "total_fights": 8,
            "last_fight_date": date(2024, 1, 1),
        }
        mock_domain_elo.return_value = {
            "striking_elo": 1500.0,
            "grappling_elo": 1500.0,
        }
        mock_features.return_value = None
        mock_accuracy.return_value = None
        mock_style_counts.return_value = {}

        session = _make_mock_session()
        result = build_matchup(session, fighter_a, fighter_b)

        assert result.style.cross_division_warning is not None
        assert "division" in result.style.cross_division_warning.lower()


class TestMatchupResultDataclass:
    """Test MatchupResult has all expected fields."""

    def test_matchup_result_fields(self):
        """MatchupResult dataclass has all expected fields."""
        result = MatchupResult(
            fighter_a_name="A",
            fighter_b_name="B",
            physical=PhysicalDifferentials(
                height_diff=3.0,
                reach_diff=2.0,
                leg_reach_diff=1.0,
                age_diff=-2.0,
                fighter_a_age=30.0,
                fighter_b_age=32.0,
            ),
            elo=EloComparison(
                fighter_a_overall=1500.0,
                fighter_b_overall=1450.0,
                fighter_a_striking=1520.0,
                fighter_b_striking=1460.0,
                fighter_a_grappling=1480.0,
                fighter_b_grappling=1490.0,
                fighter_a_division="Lightweight",
                fighter_b_division="Lightweight",
            ),
            striking=StrikingMatrix(
                a_sig_str_per_min=4.5,
                a_striking_accuracy=0.48,
                a_strike_defense=0.60,
                b_sig_str_per_min=3.8,
                b_striking_accuracy=0.42,
                b_strike_defense=0.55,
            ),
            grappling=GrapplingMatrix(
                a_td_rate=2.3,
                a_td_accuracy=0.50,
                a_td_defense=0.70,
                b_td_rate=1.8,
                b_td_accuracy=0.40,
                b_td_defense=0.60,
                a_ctrl_time=130.0,
                b_ctrl_time=100.0,
            ),
            style=StyleAnalysis(
                fighter_a_style="striker",
                fighter_b_style="grappler",
                fighter_a_index=40.0,
                fighter_b_index=-50.0,
                matchup_win_rates=[],
                cross_division_warning=None,
            ),
        )

        assert result.fighter_a_name == "A"
        assert result.physical.height_diff == 3.0
        assert result.elo.fighter_a_overall == 1500.0
        assert result.striking.a_sig_str_per_min == 4.5
        assert result.grappling.a_td_rate == 2.3
        assert result.style.fighter_a_style == "striker"
        assert result.style.cross_division_warning is None
