"""LIVE-* acceptance: closing_prob_diff populates on cold + warm cache.

Per CONTEXT.md ``<plan_specific_requirements>`` 16-02 final acceptance test
(Pitfall #4 mitigation):

  Predict an upcoming card matchup with the cache cold (forcing live fetch),
  then with the cache populated (cache hit path). Confirm closing_prob_diff
  populates in BOTH cases vs the v1.1 NaN-pad behavior.

Pre-Phase-16 baseline: ``closing_prob_diff`` was ALWAYS NaN at predict time
(predictor.py:289-352 NaN-pad block). After this plan: it's NaN only when
both BFO paths fail. The test pins both the populated-from-live and
populated-from-cache cases.

The test also asserts the D-11 observability log line is written per call,
with the correct ``odds_source`` value.

Test markers (additive, no pyproject.toml change required because pytest
warns on unrecognized markers but doesn't fail).
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ufc_prediction.ml.config import FEATURE_COLUMNS

pytestmark = [pytest.mark.integration]


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def upcoming_event_date() -> date:
    return date.today() + timedelta(days=7)


@pytest.fixture
def predict_log_path(tmp_path, monkeypatch) -> Path:
    """Redirect the predict-trace JSONL to a tmp file isolated per test."""
    log_path = tmp_path / "predict-trace.jsonl"
    monkeypatch.setenv("UFC_PREDICT_LOG", str(log_path))
    return log_path


@pytest.fixture
def stub_fighters():
    """Return two duck-typed Fighter rows that ``inference_features.build``
    can read (id, name, physicals, stance, dob)."""
    fa = MagicMock(
        id=1,
        name="Khabib Nurmagomedov",
        source="ufcstats",
        height_inches=70.0,
        reach_inches=70.0,
        leg_reach_inches=39.0,
        stance="Orthodox",
        date_of_birth=date(1988, 9, 20),
    )
    fa.name = "Khabib Nurmagomedov"
    fb = MagicMock(
        id=2,
        name="Conor McGregor",
        source="ufcstats",
        height_inches=69.0,
        reach_inches=74.0,
        leg_reach_inches=40.0,
        stance="Southpaw",
        date_of_birth=date(1988, 7, 14),
    )
    fb.name = "Conor McGregor"
    return fa, fb


@pytest.fixture
def patched_predictor(stub_fighters):
    """Build a ModelPredictor with the live model, but bypass the DB-backed
    ``_resolve_fighter`` (which would hit Postgres) and the snapshot-reading
    helpers in inference_features so the test runs without a Docker daemon.

    Returns ``(predictor_instance, fighter_a, fighter_b, mock_session)``.
    """
    from ufc_prediction.ml import inference_features as inf
    from ufc_prediction.ml import predictor as pred_mod

    if not Path("models/xgb_v2.joblib").exists():
        pytest.skip("xgb_v2.joblib not present in this worktree")

    # Patch the helpers inference_features uses so we don't need a real DB
    fa, fb = stub_fighters

    def fake_get_elo(session, fighter_id, elo_type):
        is_a = fighter_id == fa.id
        if elo_type == "overall":
            return 1620.0 if is_a else 1480.0
        if elo_type == "striking":
            return 1600.0 if is_a else 1520.0
        if elo_type == "grappling":
            return 1700.0 if is_a else 1450.0
        return 1500.0

    def fake_get_perf(session, fighter_id):
        # Provide a partial fixture so several performance _diff feats are populated
        return {
            "sig_str_per_minute": 4.5 if fighter_id == fa.id else 5.5,
            "td_rate": 3.0 if fighter_id == fa.id else 0.5,
            "td_accuracy": 0.50 if fighter_id == fa.id else 0.30,
            "ctrl_time_per_fight": 200.0 if fighter_id == fa.id else 30.0,
        }

    # _resolve_fighter is module-level in predictor.py — patch it to bypass DB
    def fake_resolve_fighter(session, name):
        return fa if name == fa.name else fb

    # ModelPredictor.__init__ runs LIVE-03 assertion against meta JSON;
    # xgb_v2's saved meta already matches FEATURE_COLUMNS, so no monkey-patch
    # is needed for the model load itself.
    predictor_instance = pred_mod.ModelPredictor(
        model_dir="models",
        version="v2",
    )

    # Patch the heavy DB readers that inference_features pulls from
    with (
        patch.object(inf, "_get_latest_elo", fake_get_elo),
        patch.object(inf, "_get_latest_computed_features", fake_get_perf),
        patch.object(pred_mod, "_resolve_fighter", fake_resolve_fighter),
        patch.object(pred_mod, "_get_latest_elo", fake_get_elo),
    ):
        yield predictor_instance, fa, fb, MagicMock()


# ── Helpers ────────────────────────────────────────────────────────────────


def _read_last_log_line(path: Path) -> dict:
    assert path.exists(), f"D-11 trace file missing: {path}"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1, f"D-11 trace file has no lines: {path}"
    return json.loads(lines[-1])


def _capture_inference_vec(monkeypatch):
    """Wrap ``inference_features.build`` so we can read the produced
    feature vector after a predict call."""
    from ufc_prediction.ml import inference_features as inf
    from ufc_prediction.ml import predictor as pred_mod

    captured = {}
    real_build = inf.build

    def wrapper(*args, **kwargs):
        vec = real_build(*args, **kwargs)
        captured["vec"] = vec
        return vec

    # The predictor imports build under an alias — patch BOTH symbols
    monkeypatch.setattr(inf, "build", wrapper)
    monkeypatch.setattr(pred_mod, "build_inference_features", wrapper)
    return captured


# ── Test 1: cold cache → live fetch populates closing_prob_diff ─────────────


def test_cold_cache_live_fetch_populates_closing_prob_diff(
    patched_predictor,
    predict_log_path,
    upcoming_event_date,
    monkeypatch,
):
    """Cache empty + live HTTP succeeds → closing_prob_diff non-NaN
    + ``odds_source == "live"`` in the JSONL trace.
    """
    from ufc_prediction.scraper import bfo_live
    from ufc_prediction.scraper.bfo_live import MatchupOdds

    pred, fa, fb, session = patched_predictor

    # Force cache miss (D-09 step 1 returns None)
    monkeypatch.setattr(bfo_live, "_try_cache", lambda *a, **kw: None)

    # Mock the live BFO fetch to return MatchupOdds(source="live") with
    # populated A/B moneylines so the 5-feature odds block computes.
    live_odds = MatchupOdds(
        fighter_a_opening=-200,
        fighter_a_closing_min=-220,
        fighter_a_closing_max=-180,
        fighter_b_opening=170,
        fighter_b_closing_min=150,
        fighter_b_closing_max=190,
        fetched_at=datetime.now(timezone.utc),
        source="live",
    )
    monkeypatch.setattr(
        bfo_live,
        "_try_live",
        lambda *a, **kw: live_odds,
    )

    captured = _capture_inference_vec(monkeypatch)

    result = pred.predict(
        session,
        fa.name,
        fb.name,
        event_date=upcoming_event_date,
    )

    # closing_prob_diff is NaN-busted: the value is finite, not NaN.
    vec = captured["vec"][0]
    cl_idx = FEATURE_COLUMNS.index("closing_prob_diff")
    assert not np.isnan(vec[cl_idx]), (
        f"closing_prob_diff still NaN under live fetch (regression on "
        f"D-12). Got {vec[cl_idx]!r}; expected finite float."
    )
    assert vec[cl_idx] > 0  # A favorite by closing line

    # Result dict reflects live source
    assert result["odds_source"] == "live"

    # D-11 trace logged correctly
    record = _read_last_log_line(predict_log_path)
    assert record["odds_source"] == "live"
    assert record["fighter_a"] == fa.name
    assert record["fighter_b"] == fb.name
    assert record["model_version"] == "v2"
    assert isinstance(record["xgb_proba_a"], float)


# ── Test 2: warm cache → no HTTP, closing_prob_diff still populated ─────────


def test_warm_cache_skips_http_populates_closing_prob_diff(
    patched_predictor,
    predict_log_path,
    upcoming_event_date,
    monkeypatch,
):
    """Cache hit → ``odds_source == "cache"``; closing_prob_diff still
    populated; the live HTTP path is never invoked.
    """
    from ufc_prediction.scraper import bfo_live
    from ufc_prediction.scraper.bfo_live import MatchupOdds

    pred, fa, fb, session = patched_predictor

    cached = MatchupOdds(
        fighter_a_opening=-180,
        fighter_a_closing_min=-200,
        fighter_a_closing_max=-160,
        fighter_b_opening=160,
        fighter_b_closing_min=140,
        fighter_b_closing_max=180,
        fetched_at=datetime.now(timezone.utc),
        source="cache",
    )

    # Cache hit
    monkeypatch.setattr(bfo_live, "_try_cache", lambda *a, **kw: cached)

    # If _try_live is called the test fails — cache should short-circuit
    def boom_live(*args, **kwargs):
        raise AssertionError("live HTTP path called on cache hit (D-09 violation)")

    monkeypatch.setattr(bfo_live, "_try_live", boom_live)

    # ALSO patch inference_features._get_cached_odds so that when build()
    # runs with live_odds=cached (from bfo_live), we bypass the inner
    # cache lookup; build() will use live_odds directly.
    captured = _capture_inference_vec(monkeypatch)

    result = pred.predict(
        session,
        fa.name,
        fb.name,
        event_date=upcoming_event_date,
    )

    vec = captured["vec"][0]
    cl_idx = FEATURE_COLUMNS.index("closing_prob_diff")
    assert not np.isnan(vec[cl_idx]), "closing_prob_diff NaN on warm cache — D-09 cache path broken"
    assert vec[cl_idx] > 0

    assert result["odds_source"] == "cache"

    record = _read_last_log_line(predict_log_path)
    assert record["odds_source"] == "cache"
    assert record["fighter_a"] == fa.name


# ── Test 3: no BFO data → predict still succeeds; odds NaN ──────────────────


def test_no_bfo_data_falls_back_to_nan(
    patched_predictor,
    predict_log_path,
    upcoming_event_date,
    monkeypatch,
):
    """Cache miss + live timeout → ``odds_source == "nan"``; closing_prob_diff
    is NaN; predict still returns a valid result (XGBoost native NaN per D-04).
    """
    from ufc_prediction.scraper import bfo_live

    pred, fa, fb, session = patched_predictor

    monkeypatch.setattr(bfo_live, "_try_cache", lambda *a, **kw: None)
    monkeypatch.setattr(bfo_live, "_try_live", lambda *a, **kw: None)

    captured = _capture_inference_vec(monkeypatch)

    result = pred.predict(
        session,
        fa.name,
        fb.name,
        event_date=upcoming_event_date,
    )

    vec = captured["vec"][0]
    cl_idx = FEATURE_COLUMNS.index("closing_prob_diff")
    assert np.isnan(vec[cl_idx])  # NaN, NEVER 0.0
    # But predict() still returned a valid probability — XGBoost handles NaN
    assert 0.0 <= result["model_probability_a"] <= 1.0

    assert result["odds_source"] == "nan"

    record = _read_last_log_line(predict_log_path)
    assert record["odds_source"] == "nan"
    assert record["odds_timestamp_iso"] is None
