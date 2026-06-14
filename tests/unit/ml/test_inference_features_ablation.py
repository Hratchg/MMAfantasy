"""Phase 18 NET-V2-01 Wave-0 RED — include_net flag through inference_features.build.

Goal: when called with `include_net=False`, build returns shape (1, 72)
(drops the trailing 3 NET-* columns). Default include_net=True preserves
backwards compat (shape (1, 75)).

Lifts `_stub_fighter` + `_patch_session_with_snapshots` VERBATIM from
test_inference_features.py:29-80 (canonical fixture).

These tests RED on import (Wave 0) — `FEATURE_COLUMNS_NO_NET` does not yet
exist in `config.py`. Goes GREEN at Wave 1 Tasks 9 + 10.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from ufc_prediction.ml.config import FEATURE_COLUMNS_NO_NET

inference_features = pytest.importorskip("ufc_prediction.ml.inference_features")


# ── Helpers (lifted verbatim from tests/unit/ml/test_inference_features.py) ──


def _stub_fighter(fighter_id: int, **kwargs):
    """Build a duck-typed fighter row with the attributes inference_features reads."""
    defaults = {
        "id": fighter_id,
        "name": f"Fighter-{fighter_id}",
        "height_inches": 72.0,
        "reach_inches": 74.0,
        "leg_reach_inches": 42.0,
        "stance": "Orthodox",
        "date_of_birth": date(1990, 1, 1),
    }
    defaults.update(kwargs)
    return MagicMock(**defaults)


def _patch_session_with_snapshots(
    monkeypatch,
    *,
    elo_a: float = 1520.0,
    elo_b: float = 1480.0,
    elo_a_striking: float = 1530.0,
    elo_b_striking: float = 1470.0,
    elo_a_grappling: float = 1510.0,
    elo_b_grappling: float = 1490.0,
    perf_a: dict | None = None,
    perf_b: dict | None = None,
    cached_odds_a: dict | None = None,
    cached_odds_b: dict | None = None,
):
    """Patch the four DB-reading helpers in inference_features."""
    perf_a_default = {key: 1.0 for key in (
        "sig_str_per_minute", "sig_str_per_minute_ewma",
        "total_str_per_minute", "total_str_per_minute_ewma",
        "td_rate", "td_rate_ewma", "td_accuracy", "td_accuracy_ewma",
        "td_defense", "td_defense_ewma", "strike_defense", "strike_defense_ewma",
        "ctrl_time_per_fight", "ctrl_time_per_fight_ewma",
        "sub_att_per_fight", "sub_att_per_fight_ewma",
        "opp_adj_sig_str", "opp_adj_td", "opp_adj_strike_def", "opp_adj_ctrl_time",
    )}
    perf_b_default = {k: 0.5 for k in perf_a_default}

    pa = perf_a if perf_a is not None else perf_a_default
    pb = perf_b if perf_b is not None else perf_b_default

    def fake_get_elo(session, fighter_id, elo_type):
        is_a = (fighter_id == 1)
        if elo_type == "overall":
            return elo_a if is_a else elo_b
        if elo_type == "striking":
            return elo_a_striking if is_a else elo_b_striking
        if elo_type == "grappling":
            return elo_a_grappling if is_a else elo_b_grappling
        return 1500.0

    def fake_get_perf(session, fighter_id):
        return pa if fighter_id == 1 else pb

    def fake_get_cached_odds(session, fa_id, fb_id, event_date):
        return cached_odds_a, cached_odds_b

    monkeypatch.setattr(inference_features, "_get_latest_elo", fake_get_elo)
    monkeypatch.setattr(
        inference_features, "_get_latest_computed_features", fake_get_perf
    )
    monkeypatch.setattr(
        inference_features, "_get_cached_odds", fake_get_cached_odds
    )
    return MagicMock()


# ── NET-V2-01 ablation tests ────────────────────────────────────────────────


def test_build_with_include_net_false_returns_72_cols(monkeypatch):
    """include_net=False → vector.shape == (1, 72)."""
    session = _patch_session_with_snapshots(monkeypatch)
    fa = _stub_fighter(1)
    fb = _stub_fighter(2)
    vec = inference_features.build(
        session, fa, fb, date(2026, 6, 14), include_net=False,
    )
    assert vec.shape == (1, 72)
    assert vec.shape[1] == len(FEATURE_COLUMNS_NO_NET)


def test_build_default_include_net_true_returns_75_cols(monkeypatch):
    """Default include_net=True → vector.shape == (1, 75) (backwards compat)."""
    session = _patch_session_with_snapshots(monkeypatch)
    fa = _stub_fighter(1)
    fb = _stub_fighter(2)
    vec = inference_features.build(session, fa, fb, date(2026, 6, 14))
    assert vec.shape == (1, 75)
