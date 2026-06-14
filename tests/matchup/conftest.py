"""Shared test fixtures for matchup tests."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest


@pytest.fixture()
def make_fighter():
    """Factory returning a Fighter-like object with common physical attributes.

    Uses SimpleNamespace to avoid needing a real DB session for unit tests.
    """

    def _make(
        *,
        name: str = "Fighter A",
        height_inches: float | None = 72.0,
        reach_inches: float | None = 74.0,
        leg_reach_inches: float | None = 42.0,
        date_of_birth: date | None = date(1990, 1, 1),
    ) -> SimpleNamespace:
        return SimpleNamespace(
            name=name,
            height_inches=height_inches,
            reach_inches=reach_inches,
            leg_reach_inches=leg_reach_inches,
            date_of_birth=date_of_birth,
        )

    return _make


@pytest.fixture()
def sample_features_full() -> dict:
    """Complete features JSON with all EWMA and career keys populated."""
    return {
        "sig_str_per_minute": 4.0,
        "sig_str_per_minute_ewma": 4.5,
        "total_str_per_minute": 5.0,
        "total_str_per_minute_ewma": 5.5,
        "td_rate": 2.0,
        "td_rate_ewma": 2.3,
        "td_accuracy": 0.45,
        "td_accuracy_ewma": 0.50,
        "td_defense": 0.65,
        "td_defense_ewma": 0.70,
        "strike_defense": 0.55,
        "strike_defense_ewma": 0.60,
        "ctrl_time_per_fight": 120.0,
        "ctrl_time_per_fight_ewma": 130.0,
        "sub_att_per_fight": 0.5,
        "sub_att_per_fight_ewma": 0.6,
        "opp_adj_sig_str": 3.8,
        "opp_adj_td": 1.9,
        "opp_adj_strike_def": 0.58,
        "opp_adj_ctrl_time": 100.0,
        "style_tag": "striker",
    }


@pytest.fixture()
def sample_features_partial() -> dict:
    """Features JSON with only career keys (no EWMA) for fallback testing."""
    return {
        "sig_str_per_minute": 4.0,
        "total_str_per_minute": 5.0,
        "td_rate": 2.0,
        "td_accuracy": 0.45,
        "td_defense": 0.65,
        "strike_defense": 0.55,
        "ctrl_time_per_fight": 120.0,
        "sub_att_per_fight": 0.5,
        "opp_adj_sig_str": 3.8,
        "opp_adj_td": 1.9,
        "opp_adj_strike_def": 0.58,
        "opp_adj_ctrl_time": 100.0,
        "style_tag": "balanced",
    }
