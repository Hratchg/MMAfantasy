"""Phase 18 NET-V2-01 Wave-0 RED — include_net flag through FeatureMatrixAssembler.

Goal: when assembled with `include_net=False`, the output matrix has
72 columns (drops the trailing 3 NET-* features). Default include_net=True
preserves backwards compat (75-col output).

These tests RED on import (Wave 0) — `FEATURE_COLUMNS_NO_NET` does not yet
exist in `config.py`. Goes GREEN when Wave 1 Tasks 9 + 10 land.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from ufc_prediction.ml.config import FEATURE_COLUMNS, FEATURE_COLUMNS_NO_NET
from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler


def _toy_fight_records() -> list[dict]:
    """Two-fight chronological fight list (mirrors test_inference_features patterns)."""
    return [
        {
            "fight_id": 1,
            "event_date": date(2020, 1, 1),
            "fighter_a_id": 1,
            "fighter_b_id": 2,
            "winner_id": 1,
            "loser_id": 2,
            "method": "Decision",
            "weight_class": "Lightweight",
        },
        {
            "fight_id": 2,
            "event_date": date(2020, 6, 1),
            "fighter_a_id": 2,
            "fighter_b_id": 1,
            "winner_id": 2,
            "loser_id": 1,
            "method": "KO/TKO",
            "weight_class": "Lightweight",
        },
    ]


def _toy_elo_features() -> dict[tuple[int, int], dict[str, float]]:
    """Per-fight pre-fight Elo snapshots."""
    return {
        (1, 1): {"elo_overall": 1520.0, "elo_striking": 1530.0, "elo_grappling": 1510.0},
        (1, 2): {"elo_overall": 1480.0, "elo_striking": 1470.0, "elo_grappling": 1490.0},
        (2, 1): {"elo_overall": 1530.0, "elo_striking": 1540.0, "elo_grappling": 1520.0},
        (2, 2): {"elo_overall": 1470.0, "elo_striking": 1460.0, "elo_grappling": 1480.0},
    }


def _toy_computed_features() -> dict[tuple[int, int], dict[str, float | None]]:
    """Per-fight pre-fight career-stat snapshots; populate keys both perf
    and NET-* expects so assembler can emit a complete row."""
    base = {
        "sig_str_per_minute": 4.5,
        "sig_str_per_minute_ewma": 4.4,
        "total_str_per_minute": 6.0,
        "total_str_per_minute_ewma": 5.9,
        "td_rate": 1.5,
        "td_rate_ewma": 1.4,
        "td_accuracy": 0.45,
        "td_accuracy_ewma": 0.44,
        "td_defense": 0.65,
        "td_defense_ewma": 0.64,
        "strike_defense": 0.55,
        "strike_defense_ewma": 0.54,
        "ctrl_time_per_fight": 60.0,
        "ctrl_time_per_fight_ewma": 58.0,
        "sub_att_per_fight": 0.5,
        "sub_att_per_fight_ewma": 0.4,
        "opp_adj_sig_str": 0.0,
        "opp_adj_td": 0.0,
        "opp_adj_strike_def": 0.0,
        "opp_adj_ctrl_time": 0.0,
        "pagerank": 0.05,
        "sos_2hop": 0.04,
        "is_debutant_in_graph": 0.0,
    }
    return {
        (1, 1): dict(base),
        (1, 2): dict(base),
        (2, 1): dict(base),
        (2, 2): dict(base),
    }


def _toy_fighter_physicals() -> dict[int, dict]:
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
            "reach_inches": 71.0,
            "leg_reach_inches": 40.0,
            "stance": "Southpaw",
            "date_of_birth": date(1991, 1, 1),
        },
    }


def _toy_division_medians() -> dict[str, dict[str, float]]:
    return {
        "Lightweight": {
            "sig_str_per_minute": 4.0,
            "td_rate": 1.5,
            "td_accuracy": 0.4,
            "td_defense": 0.6,
            "strike_defense": 0.5,
            "ctrl_time_per_fight": 50.0,
            "sub_att_per_fight": 0.3,
        },
    }


def _build_assembler() -> FeatureMatrixAssembler:
    return FeatureMatrixAssembler()


def test_assemble_with_include_net_true_returns_75_cols():
    """Default include_net=True → matrix.shape[1] == 75."""
    a = _build_assembler()
    X, _y, _dates = a.assemble(
        fight_records=_toy_fight_records(),
        elo_features=_toy_elo_features(),
        computed_features=_toy_computed_features(),
        fighter_physicals=_toy_fighter_physicals(),
        division_medians=_toy_division_medians(),
        include_net=True,
    )
    assert X.shape[1] == 75
    assert X.shape[1] == len(FEATURE_COLUMNS)


def test_assemble_with_include_net_false_returns_72_cols():
    """include_net=False → matrix.shape[1] == 72."""
    a = _build_assembler()
    X, _y, _dates = a.assemble(
        fight_records=_toy_fight_records(),
        elo_features=_toy_elo_features(),
        computed_features=_toy_computed_features(),
        fighter_physicals=_toy_fighter_physicals(),
        division_medians=_toy_division_medians(),
        include_net=False,
    )
    assert X.shape[1] == 72
    assert X.shape[1] == len(FEATURE_COLUMNS_NO_NET)


def test_assemble_default_is_75_cols_backwards_compat():
    """Calling assemble() without include_net (legacy callers) → 75 cols.

    Backwards-compat guarantee — unrelated callers (retrain scripts) keep
    working without modification.
    """
    a = _build_assembler()
    X, _y, _dates = a.assemble(
        fight_records=_toy_fight_records(),
        elo_features=_toy_elo_features(),
        computed_features=_toy_computed_features(),
        fighter_physicals=_toy_fighter_physicals(),
        division_medians=_toy_division_medians(),
    )
    assert X.shape[1] == 75
