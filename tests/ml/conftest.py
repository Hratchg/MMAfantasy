"""Shared fixtures for ML tests.

Pure Python fixtures (no DB dependency) for unit testing
the feature matrix assembly, config, and query processing logic.
"""

from __future__ import annotations

from datetime import date

import pytest


@pytest.fixture()
def fighter_physicals() -> dict[int, dict]:
    """Synthetic fighter physical data, keyed by fighter_id.

    Includes fighters with complete data, partial data (missing reach),
    and completely missing physical data.
    """
    return {
        1: {
            "height_inches": 72.0,
            "reach_inches": 74.0,
            "leg_reach_inches": 42.0,
            "stance": "Orthodox",
            "date_of_birth": date(1990, 1, 1),
        },
        2: {
            "height_inches": 70.0,
            "reach_inches": 72.0,
            "leg_reach_inches": 40.0,
            "stance": "Southpaw",
            "date_of_birth": date(1988, 6, 15),
        },
        3: {
            "height_inches": None,
            "reach_inches": None,
            "leg_reach_inches": None,
            "stance": "Switch",
            "date_of_birth": date(1992, 3, 10),
        },
        4: {
            "height_inches": 68.0,
            "reach_inches": None,
            "leg_reach_inches": 39.0,
            "stance": None,
            "date_of_birth": None,
        },
        5: {
            "height_inches": 71.0,
            "reach_inches": 73.0,
            "leg_reach_inches": 41.0,
            "stance": "Orthodox",
            "date_of_birth": date(1991, 5, 20),
        },
    }


@pytest.fixture()
def fight_records() -> list[dict]:
    """Synthetic fight records sorted chronologically.

    Covers pre-cutoff and post-cutoff fights for temporal split testing.
    All have winner_id (no draws/NCs -- those are skipped by load_fight_records).
    """
    return [
        {
            "fight_id": 101,
            "event_date": date(2020, 1, 15),
            "fighter_a_id": 1,
            "fighter_b_id": 2,
            "winner_id": 1,
            "weight_class": "Lightweight",
        },
        {
            "fight_id": 102,
            "event_date": date(2021, 6, 20),
            "fighter_a_id": 3,
            "fighter_b_id": 4,
            "winner_id": 4,
            "weight_class": "Welterweight",
        },
        {
            "fight_id": 103,
            "event_date": date(2022, 11, 5),
            "fighter_a_id": 1,
            "fighter_b_id": 5,
            "winner_id": 5,
            "weight_class": "Lightweight",
        },
        {
            "fight_id": 104,
            "event_date": date(2023, 3, 10),
            "fighter_a_id": 2,
            "fighter_b_id": 5,
            "winner_id": 2,
            "weight_class": "Lightweight",
        },
        {
            "fight_id": 105,
            "event_date": date(2024, 1, 20),
            "fighter_a_id": 1,
            "fighter_b_id": 3,
            "winner_id": 1,
            "weight_class": "Lightweight",
        },
    ]


@pytest.fixture()
def elo_features() -> dict[tuple[int, int], dict[str, float]]:
    """Synthetic Elo features keyed by (fighter_id, fight_id).

    Uses elo_before values (NOT elo_after) per ML-06.
    """
    return {
        (1, 101): {"elo_overall": 1520.0, "elo_striking": 1530.0, "elo_grappling": 1510.0},
        (2, 101): {"elo_overall": 1480.0, "elo_striking": 1470.0, "elo_grappling": 1490.0},
        (3, 102): {"elo_overall": 1500.0, "elo_striking": 1500.0, "elo_grappling": 1500.0},
        (4, 102): {"elo_overall": 1510.0, "elo_striking": 1505.0, "elo_grappling": 1515.0},
        (1, 103): {"elo_overall": 1540.0, "elo_striking": 1550.0, "elo_grappling": 1520.0},
        (5, 103): {"elo_overall": 1525.0, "elo_striking": 1520.0, "elo_grappling": 1530.0},
        (2, 104): {"elo_overall": 1500.0, "elo_striking": 1490.0, "elo_grappling": 1510.0},
        (5, 104): {"elo_overall": 1535.0, "elo_striking": 1530.0, "elo_grappling": 1540.0},
        (1, 105): {"elo_overall": 1550.0, "elo_striking": 1560.0, "elo_grappling": 1530.0},
        (3, 105): {"elo_overall": 1505.0, "elo_striking": 1510.0, "elo_grappling": 1500.0},
    }


@pytest.fixture()
def computed_features() -> dict[tuple[int, int], dict[str, float | None]]:
    """Synthetic computed features keyed by (fighter_id, fight_id).

    Contains the 20 numeric features from CANONICAL_FEATURE_ORDER
    (excluding style_tag). Some fighters have None values for certain features.
    """
    base_features = {
        "sig_str_per_minute": 4.5,
        "sig_str_per_minute_ewma": 4.2,
        "total_str_per_minute": 6.0,
        "total_str_per_minute_ewma": 5.8,
        "td_rate": 2.5,
        "td_rate_ewma": 2.3,
        "td_accuracy": 0.45,
        "td_accuracy_ewma": 0.42,
        "td_defense": 0.65,
        "td_defense_ewma": 0.63,
        "strike_defense": 0.55,
        "strike_defense_ewma": 0.53,
        "ctrl_time_per_fight": 120.0,
        "ctrl_time_per_fight_ewma": 115.0,
        "sub_att_per_fight": 0.8,
        "sub_att_per_fight_ewma": 0.75,
        "opp_adj_sig_str": 3.8,
        "opp_adj_td": 1.9,
        "opp_adj_strike_def": 0.5,
        "opp_adj_ctrl_time": 100.0,
    }

    alt_features = {
        "sig_str_per_minute": 3.5,
        "sig_str_per_minute_ewma": 3.3,
        "total_str_per_minute": 5.0,
        "total_str_per_minute_ewma": 4.8,
        "td_rate": 3.0,
        "td_rate_ewma": 2.8,
        "td_accuracy": 0.50,
        "td_accuracy_ewma": 0.48,
        "td_defense": 0.60,
        "td_defense_ewma": 0.58,
        "strike_defense": 0.50,
        "strike_defense_ewma": 0.48,
        "ctrl_time_per_fight": 140.0,
        "ctrl_time_per_fight_ewma": 135.0,
        "sub_att_per_fight": 1.2,
        "sub_att_per_fight_ewma": 1.1,
        "opp_adj_sig_str": 3.2,
        "opp_adj_td": 2.4,
        "opp_adj_strike_def": 0.45,
        "opp_adj_ctrl_time": 110.0,
    }

    return {
        (1, 101): dict(base_features),
        (2, 101): dict(alt_features),
        (3, 102): dict(base_features),
        (4, 102): dict(alt_features),
        (1, 103): dict(base_features),
        (5, 103): dict(alt_features),
        (2, 104): dict(alt_features),
        (5, 104): dict(base_features),
        (1, 105): dict(base_features),
        (3, 105): dict(alt_features),
    }


@pytest.fixture()
def division_medians() -> dict[str, dict[str, float]]:
    """Pre-computed division medians for imputation testing."""
    return {
        "Lightweight": {
            "height_inches": 71.0,
            "reach_inches": 73.0,
            "leg_reach_inches": 41.0,
        },
        "Welterweight": {
            "height_inches": 72.0,
            "reach_inches": 74.0,
            "leg_reach_inches": 42.0,
        },
    }
