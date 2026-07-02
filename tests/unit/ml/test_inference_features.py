"""Tests for ufc_prediction.ml.inference_features (LIVE-02).

Predict-time twin of feature_matrix.py. Closes the 44/72 NaN-pad gap at
predictor.py:289-352 (deleted in Task 4 of this plan per D-12).

Per Gotcha 2: the 5-feature Phase 15.1 odds block at feature_matrix.py:560-625
MUST be replicated column-for-column. Test 3 / 4 / 5 are the parity gates.
Per Pattern D: NaN — never 0.0 — for missing odds.
Per Pitfall #12: column ordering is locked by FEATURE_COLUMNS.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from ufc_prediction.ml.config import FEATURE_COLUMNS

inference_features = pytest.importorskip("ufc_prediction.ml.inference_features")


# ── Helpers ─────────────────────────────────────────────────────────────────


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
    """Patch the four DB-reading helpers in inference_features so we don't
    need a real SQLAlchemy session.
    """
    perf_a_default = {
        key: 1.0
        for key in (
            "sig_str_per_minute",
            "sig_str_per_minute_ewma",
            "total_str_per_minute",
            "total_str_per_minute_ewma",
            "td_rate",
            "td_rate_ewma",
            "td_accuracy",
            "td_accuracy_ewma",
            "td_defense",
            "td_defense_ewma",
            "strike_defense",
            "strike_defense_ewma",
            "ctrl_time_per_fight",
            "ctrl_time_per_fight_ewma",
            "sub_att_per_fight",
            "sub_att_per_fight_ewma",
            "opp_adj_sig_str",
            "opp_adj_td",
            "opp_adj_strike_def",
            "opp_adj_ctrl_time",
        )
    }
    perf_b_default = {k: 0.5 for k in perf_a_default}

    pa = perf_a if perf_a is not None else perf_a_default
    pb = perf_b if perf_b is not None else perf_b_default

    def fake_get_elo(session, fighter_id, elo_type):
        is_a = fighter_id == 1
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
    monkeypatch.setattr(inference_features, "_get_latest_computed_features", fake_get_perf)
    monkeypatch.setattr(inference_features, "_get_cached_odds", fake_get_cached_odds)
    return MagicMock()


# ── Test 1: cardinality ─────────────────────────────────────────────────────


def test_inference_features_shape_matches_feature_columns(monkeypatch):
    """Test 1: build returns array of shape (1, len(FEATURE_COLUMNS))."""
    session = _patch_session_with_snapshots(monkeypatch)
    fa = _stub_fighter(1)
    fb = _stub_fighter(2)

    vec = inference_features.build(session, fa, fb, date(2026, 6, 14))

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (1, len(FEATURE_COLUMNS))
    assert vec.dtype == np.float64


# ── Test 2: NaN tolerance (relaxed per RESEARCH.md acknowledgment) ──────────


def test_inference_features_nan_tolerance_under_full_snapshot(monkeypatch):
    """Test 2: critical features non-NaN under fully-populated snapshot.

    Many career-snapshot features (pace decay, rolling windows, rematch flag)
    require fight history we don't reconstruct at predict time without a
    fight_records replay. Phase 16 ships odds + Elo + physical + performance
    + stance; the rest stays NaN at predict time and XGBoost handles it via
    sparsity-aware split finding (D-04(P14)).
    """
    session = _patch_session_with_snapshots(
        monkeypatch,
        cached_odds_a={
            "opening_implied_prob": 0.55,
            "closing_implied_prob": 0.60,
        },
        cached_odds_b={
            "opening_implied_prob": 0.45,
            "closing_implied_prob": 0.40,
        },
    )
    fa = _stub_fighter(1)
    fb = _stub_fighter(2)

    vec = inference_features.build(session, fa, fb, date(2026, 6, 14))
    # Critical features MUST be non-NaN
    for col in (
        "elo_overall_diff",
        "elo_striking_diff",
        "elo_grappling_diff",
        "stance_matchup",
        "height_diff",
        "reach_diff",
        "age_diff",
        "closing_prob_diff",
        "opening_prob_diff",
        "sig_str_per_minute_diff",
    ):
        idx = FEATURE_COLUMNS.index(col)
        assert not np.isnan(vec[0, idx]), f"{col} unexpectedly NaN"


# ── Test 3: 5-feature odds parity with cached BFO ───────────────────────────


def test_odds_block_parity_with_cache(monkeypatch):
    """Test 3 (Gotcha 2): cached odds populate the 5-column odds block
    EXACTLY as feature_matrix.py:560-625 would.
    """
    op_a, cl_a = 0.62, 0.65
    op_b, cl_b = 0.38, 0.35
    elo_a, elo_b = 1520.0, 1480.0

    session = _patch_session_with_snapshots(
        monkeypatch,
        elo_a=elo_a,
        elo_b=elo_b,
        cached_odds_a={
            "opening_implied_prob": op_a,
            "closing_implied_prob": cl_a,
        },
        cached_odds_b={
            "opening_implied_prob": op_b,
            "closing_implied_prob": cl_b,
        },
    )
    fa = _stub_fighter(1)
    fb = _stub_fighter(2)

    vec = inference_features.build(session, fa, fb, date(2026, 6, 14))[0]

    # Reproduce feature_matrix.py:560-625 derivation exactly:
    expected_opening = op_a - op_b
    expected_closing = cl_a - cl_b
    expected_line_move = (cl_a - op_a) - (cl_b - op_b)
    expected_sharp = abs(expected_line_move)
    elo_diff = elo_a - elo_b
    elo_prob_a = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
    expected_divergence = cl_a - elo_prob_a

    assert math.isclose(
        vec[FEATURE_COLUMNS.index("opening_prob_diff")],
        expected_opening,
        abs_tol=1e-9,
    )
    assert math.isclose(
        vec[FEATURE_COLUMNS.index("closing_prob_diff")],
        expected_closing,
        abs_tol=1e-9,
    )
    assert math.isclose(
        vec[FEATURE_COLUMNS.index("line_movement_diff")],
        expected_line_move,
        abs_tol=1e-9,
    )
    assert math.isclose(
        vec[FEATURE_COLUMNS.index("sharp_money_signal")],
        expected_sharp,
        abs_tol=1e-9,
    )
    assert math.isclose(
        vec[FEATURE_COLUMNS.index("odds_elo_divergence")],
        expected_divergence,
        abs_tol=1e-9,
    )


# ── Test 4: 5-feature odds parity with live BFO ─────────────────────────────


def test_odds_block_parity_with_live(monkeypatch):
    """Test 4: live_odds=MatchupOdds(...) → 5-column odds block populated
    via vig-removal of the live A/B American moneylines.
    """
    from ufc_prediction.scraper.bfo_live import MatchupOdds

    live = MatchupOdds(
        fighter_a_opening=-200,
        fighter_a_closing_min=-220,
        fighter_a_closing_max=-180,
        fighter_b_opening=170,
        fighter_b_closing_min=150,
        fighter_b_closing_max=190,
        fetched_at=datetime.now(UTC),
        source="live",
    )
    session = _patch_session_with_snapshots(monkeypatch)
    fa = _stub_fighter(1)
    fb = _stub_fighter(2)

    vec = inference_features.build(
        session,
        fa,
        fb,
        date(2026, 6, 14),
        live_odds=live,
    )[0]

    op_idx = FEATURE_COLUMNS.index("opening_prob_diff")
    cl_idx = FEATURE_COLUMNS.index("closing_prob_diff")
    sharp_idx = FEATURE_COLUMNS.index("sharp_money_signal")
    div_idx = FEATURE_COLUMNS.index("odds_elo_divergence")

    # All five odds features are populated
    assert not np.isnan(vec[op_idx])
    assert not np.isnan(vec[cl_idx])
    assert not np.isnan(vec[sharp_idx])
    assert not np.isnan(vec[div_idx])
    # opening_prob_diff is positive because A is the favorite (-200 < 0)
    assert vec[op_idx] > 0
    # closing_prob_diff is positive because closing range midpoint -200 vs 170 still favors A
    assert vec[cl_idx] > 0


def test_odds_block_degrades_on_bad_live_moneyline(monkeypatch):
    """review #12 (Minor): a corrupt LIVE moneyline (|ml| < 100) must leave the
    odds features NaN and must NOT raise out of build()/predict() (Pattern D).
    """
    from ufc_prediction.scraper.bfo_live import MatchupOdds

    live = MatchupOdds(
        fighter_a_opening=-200,
        fighter_a_closing_min=-50,  # invalid: |ml| < 100 → devig raises, caught + degraded
        fighter_a_closing_max=-40,
        fighter_b_opening=170,
        fighter_b_closing_min=150,
        fighter_b_closing_max=190,
        fetched_at=datetime.now(UTC),
        source="live",
    )
    session = _patch_session_with_snapshots(monkeypatch)
    fa = _stub_fighter(1)
    fb = _stub_fighter(2)

    # Must not raise despite the bad closing moneyline.
    vec = inference_features.build(session, fa, fb, date(2026, 6, 14), live_odds=live)[0]

    cl_idx = FEATURE_COLUMNS.index("closing_prob_diff")
    op_idx = FEATURE_COLUMNS.index("opening_prob_diff")
    # Closing degrades to NaN; opening (valid MLs) is still populated.
    assert np.isnan(vec[cl_idx])
    assert not np.isnan(vec[op_idx])


# ── Test 5: no-odds fallback (NaN, never 0.0) ───────────────────────────────


def test_no_odds_fallback_is_nan_not_zero(monkeypatch):
    """Test 5 (Pitfall #3 / Pattern D): cache miss + live_odds=None →
    odds-block columns are NaN, NEVER 0.0.

    0.0 is a legal probability value (mathematical pickem); it MUST remain
    distinguishable from missing-data NaN — XGBoost's sparsity-aware split
    finding learned a default direction during training assuming NaN means
    "missing", not "pickem".
    """
    session = _patch_session_with_snapshots(
        monkeypatch,
        cached_odds_a=None,
        cached_odds_b=None,
    )
    fa = _stub_fighter(1)
    fb = _stub_fighter(2)

    vec = inference_features.build(
        session,
        fa,
        fb,
        date(2026, 6, 14),
        live_odds=None,
    )[0]

    for col in (
        "opening_prob_diff",
        "closing_prob_diff",
        "line_movement_diff",
        "sharp_money_signal",
        "odds_elo_divergence",
    ):
        idx = FEATURE_COLUMNS.index(col)
        assert np.isnan(vec[idx]), f"{col} should be NaN, got {vec[idx]}"


# ── Test 6: column order locked ─────────────────────────────────────────────


def test_column_order_matches_feature_columns(monkeypatch):
    """Test 6: positional ordering of returned vector matches FEATURE_COLUMNS.

    The xgb_v2.predict_proba contract is positional — silent column-order
    drift is a Pitfall #12 silent-garbage-prediction regression.
    """
    session = _patch_session_with_snapshots(monkeypatch)
    fa = _stub_fighter(1)
    fb = _stub_fighter(2)

    vec = inference_features.build(session, fa, fb, date(2026, 6, 14))[0]

    # elo_overall_diff at position 0 (first column in FEATURE_COLUMNS)
    assert FEATURE_COLUMNS[0] == "elo_overall_diff"
    assert math.isclose(vec[0], 1520.0 - 1480.0)  # from default snapshot fixture

    # stance_matchup is binary in {0.0, 1.0}; must be at the documented index
    sm_idx = FEATURE_COLUMNS.index("stance_matchup")
    assert vec[sm_idx] in (0.0, 1.0)


# ── Test 7: debutant tolerant ───────────────────────────────────────────────


def test_debutant_with_no_snapshot_data_still_returns_shape(monkeypatch):
    """Test 7: both fighters have NO prior snapshots (debutants on both sides)
    → vector shape preserved; mostly NaN; no exception.
    """
    monkeypatch.setattr(inference_features, "_get_latest_elo", lambda *a, **kw: 1500.0)
    monkeypatch.setattr(
        inference_features,
        "_get_latest_computed_features",
        lambda *a, **kw: {},
    )
    monkeypatch.setattr(
        inference_features,
        "_get_cached_odds",
        lambda *a, **kw: (None, None),
    )

    fa = _stub_fighter(1)
    fb = _stub_fighter(2)

    vec = inference_features.build(MagicMock(), fa, fb, date(2026, 6, 14))
    assert vec.shape == (1, len(FEATURE_COLUMNS))


# ── Test 8: unknown_keys guard ──────────────────────────────────────────────


def test_unknown_keys_guard_raises(monkeypatch):
    """Test 8: if a populator emits a key not in FEATURE_COLUMNS,
    build() raises RuntimeError (lifted from predictor.py:344-352).
    """
    session = _patch_session_with_snapshots(monkeypatch)
    fa = _stub_fighter(1)
    fb = _stub_fighter(2)

    real_pop = inference_features._populate_odds

    def bad_populator(*args, **kwargs):
        feats = args[0]
        real_pop(*args, **kwargs)
        feats["this_key_is_not_in_feature_columns"] = 1.0

    monkeypatch.setattr(inference_features, "_populate_odds", bad_populator)

    with pytest.raises(RuntimeError, match="not in model feature_columns"):
        inference_features.build(session, fa, fb, date(2026, 6, 14))


# ── TestV22Dispatch — Phase 23 Plan 23-01 Task 2 ────────────────────────────


class TestV22Dispatch:
    """Phase 23 Plan 23-01 Task 2 — feature_set='v2.2' dispatch in build().

    Tests the live-prediction path's v2.2 wiring. The cross-path parity test
    (train ⇔ inference REF cols byte-identical) lives in
    tests/unit/ml/test_features_v22.py::TestTrainInferenceParity.
    """

    def _stub_meta_queries(self, monkeypatch) -> None:
        """Patch all 5 META DB-query helpers so MagicMock session is safe.

        Phase 23 Plan 23-03: META adds 5 DB-query helpers that the bare-bones
        MagicMock session in these tests can't satisfy. Stub them to return
        empty/safe defaults (debutant fighter, no prior fights, no division).
        """
        monkeypatch.setattr(
            inference_features,
            "_query_fighter_prior_fight_date",
            lambda s, fid, bd: None,
        )
        monkeypatch.setattr(
            inference_features,
            "_query_elo_history",
            lambda s, fid, bd, limit=6: [],
        )
        monkeypatch.setattr(
            inference_features,
            "_query_division_state",
            lambda s, wc, bd: ({}, 0.0),
        )
        monkeypatch.setattr(
            inference_features,
            "_query_division_mean_reach",
            lambda s, wc: None,
        )
        monkeypatch.setattr(
            inference_features,
            "_query_fighter_division",
            lambda s, fid: None,
        )

    def test_inference_v22_dispatch_shape(self, monkeypatch) -> None:
        """Test 6: build(..., feature_set='v2.2') → row length == 90."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS_V22

        session = _patch_session_with_snapshots(monkeypatch)
        # Stub _query_ref_state so we don't hit a real DB.
        monkeypatch.setattr(
            inference_features,
            "_query_ref_state",
            lambda s, r, d: ({}, {"finish": 0.5, "decision": 0.4, "no_action": 0.1}),
        )
        self._stub_meta_queries(monkeypatch)
        fa = _stub_fighter(1)
        fb = _stub_fighter(2)

        vec = inference_features.build(
            session,
            fa,
            fb,
            date(2026, 6, 14),
            feature_set="v2.2",
        )

        assert vec.shape == (1, len(FEATURE_COLUMNS_V22))
        assert vec.shape[1] == 90

    def test_inference_v21_dispatch_unchanged(self, monkeypatch) -> None:
        """Test 7: include_net=False back-compat shim still returns 72 cols."""
        session = _patch_session_with_snapshots(monkeypatch)
        fa = _stub_fighter(1)
        fb = _stub_fighter(2)

        vec = inference_features.build(
            session,
            fa,
            fb,
            date(2026, 6, 14),
            include_net=False,
        )
        assert vec.shape == (1, 72)

    def test_inference_v22_ref_cols_unknown_referee_collapses(
        self,
        monkeypatch,
    ) -> None:
        """referee_id=None → all 3 REF cols == global_rates."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS_V22

        global_rates = {"finish": 0.55, "decision": 0.35, "no_action": 0.10}
        session = _patch_session_with_snapshots(monkeypatch)
        monkeypatch.setattr(
            inference_features,
            "_query_ref_state",
            lambda s, r, d: ({}, global_rates),
        )
        self._stub_meta_queries(monkeypatch)
        fa = _stub_fighter(1)
        fb = _stub_fighter(2)

        vec = inference_features.build(
            session,
            fa,
            fb,
            date(2026, 6, 14),
            feature_set="v2.2",
            referee_id=None,
        )[0]

        idx_finish = FEATURE_COLUMNS_V22.index("ref_finish_rate_shrunk")
        idx_decision = FEATURE_COLUMNS_V22.index("ref_decision_rate_shrunk")
        idx_no_action = FEATURE_COLUMNS_V22.index("ref_no_action_rate_shrunk")
        assert math.isclose(vec[idx_finish], global_rates["finish"], abs_tol=1e-9)
        assert math.isclose(vec[idx_decision], global_rates["decision"], abs_tol=1e-9)
        assert math.isclose(vec[idx_no_action], global_rates["no_action"], abs_tol=1e-9)

    def test_inference_v22_ref_cols_with_history(self, monkeypatch) -> None:
        """referee with prior history → REF cols match shared-helper output."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS_V22
        from ufc_prediction.ml.features_v22.ref import compute_ref_rates_shrunk

        # Pre-fight history for ref 7: one finish, one decision before
        # event_date=2026-06-14.
        ref_history = {
            7: [
                {"event_date": date(2025, 1, 1), "method": "KO/TKO"},
                {"event_date": date(2025, 6, 1), "method": "Decision - Unanimous"},
            ]
        }
        global_rates = {"finish": 0.50, "decision": 0.40, "no_action": 0.10}

        session = _patch_session_with_snapshots(monkeypatch)
        monkeypatch.setattr(
            inference_features,
            "_query_ref_state",
            lambda s, r, d: (ref_history, global_rates),
        )
        self._stub_meta_queries(monkeypatch)
        fa = _stub_fighter(1)
        fb = _stub_fighter(2)

        vec = inference_features.build(
            session,
            fa,
            fb,
            date(2026, 6, 14),
            feature_set="v2.2",
            referee_id=7,
        )[0]

        expected = compute_ref_rates_shrunk(
            7,
            date(2026, 6, 14),
            ref_history,
            global_rates,
        )
        idx_finish = FEATURE_COLUMNS_V22.index("ref_finish_rate_shrunk")
        idx_decision = FEATURE_COLUMNS_V22.index("ref_decision_rate_shrunk")
        idx_no_action = FEATURE_COLUMNS_V22.index("ref_no_action_rate_shrunk")
        assert math.isclose(
            vec[idx_finish],
            expected["ref_finish_rate_shrunk"],
            abs_tol=1e-12,
        )
        assert math.isclose(
            vec[idx_decision],
            expected["ref_decision_rate_shrunk"],
            abs_tol=1e-12,
        )
        assert math.isclose(
            vec[idx_no_action],
            expected["ref_no_action_rate_shrunk"],
            abs_tol=1e-12,
        )

    def test_inference_v22_meta_cols_populated_debutant_fighters(
        self,
        monkeypatch,
    ) -> None:
        """Phase 23 Plan 23-03: META cols (81-89) NOW POPULATED.

        With debutant fighters (no prior fights, no division history, no
        division mean reach):
            - layoff_days_red/blue = 0.0 (first-fight sentinel)
            - age_at_fight_red/blue = computed from DoB (non-NaN)
            - elo_*_velocity_diff = NaN (insufficient history < 6 fights)
            - division_finish_rate_shrunk = NaN-or-0 (no class → 0 fallback)
            - reach_diff_normalized = NaN (mean_reach=0 → div-by-zero guard)

        TRAVEL (75-80) stays NaN because no event_id is provided.
        """
        from ufc_prediction.ml.config import FEATURE_COLUMNS_V22

        session = _patch_session_with_snapshots(monkeypatch)
        monkeypatch.setattr(
            inference_features,
            "_query_ref_state",
            lambda s, r, d: ({}, {"finish": 0.5, "decision": 0.4, "no_action": 0.1}),
        )
        self._stub_meta_queries(monkeypatch)
        fa = _stub_fighter(1)
        fb = _stub_fighter(2)

        vec = inference_features.build(
            session,
            fa,
            fb,
            date(2026, 6, 14),
            feature_set="v2.2",
        )[0]

        # Layoff cols (81-82): 0.0 sentinel for debutants (prior_fight_date=None).
        idx_layoff_red = FEATURE_COLUMNS_V22.index("layoff_days_red")
        idx_layoff_blue = FEATURE_COLUMNS_V22.index("layoff_days_blue")
        assert vec[idx_layoff_red] == 0.0, (
            f"layoff_days_red should be 0.0 sentinel for debutant; got {vec[idx_layoff_red]}"
        )
        assert vec[idx_layoff_blue] == 0.0

        # Age cols (83-84): non-NaN because fighter DoB is set.
        idx_age_red = FEATURE_COLUMNS_V22.index("age_at_fight_red")
        idx_age_blue = FEATURE_COLUMNS_V22.index("age_at_fight_blue")
        assert not np.isnan(vec[idx_age_red]), (
            "age_at_fight_red unexpectedly NaN; DoB stub is 1990-01-01"
        )
        assert not np.isnan(vec[idx_age_blue])

        # Elo velocity cols (85-87): NaN (insufficient history for debutant).
        for col in (
            "elo_overall_velocity_diff",
            "elo_striking_velocity_diff",
            "elo_grappling_velocity_diff",
        ):
            idx = FEATURE_COLUMNS_V22.index(col)
            assert np.isnan(vec[idx]), (
                f"{col} should be NaN for debutant (insufficient history); got {vec[idx]}"
            )

        # Reach normalized (89): NaN because mean_reach=0 (no division data).
        idx_reach = FEATURE_COLUMNS_V22.index("reach_diff_normalized")
        assert np.isnan(vec[idx_reach]), (
            f"reach_diff_normalized should NaN when division mean reach is 0; got {vec[idx_reach]}"
        )

        # cols 75..80 are TRAVEL — NaN here because event_id is None
        # (graceful degradation, NOT sentinel 0).
        for idx in range(75, 81):
            assert np.isnan(vec[idx]), (
                f"col {idx} should NaN-degrade when event_id is None; got {vec[idx]}"
            )

    def test_load_time_assert_v22_columns(self) -> None:
        """Test 9: inference_features defines _EXPECTED_V22_NCOLS == 90 (D-10)."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS_V22

        assert len(FEATURE_COLUMNS_V22) == inference_features._EXPECTED_V22_NCOLS
        assert inference_features._EXPECTED_V22_NCOLS == 90


# ── TestV22Travel — Phase 23 Plan 23-02 ─────────────────────────────────────


class TestV22Travel:
    """Phase 23 Plan 23-02 — TRAVEL cols (75-80) wired via shared helper.

    Stubs `_query_current_venue` + `_query_fighter_prior_venue` to avoid
    real DB calls; the math goes through `compute_travel_features` from
    features_v22.travel — same helper feature_matrix.py uses (Pitfall #12).
    """

    def _stub_travel_queries(
        self,
        monkeypatch,
        curr_venue: dict | None,
        prior_a: dict | None,
        prior_b: dict | None,
    ) -> None:
        """Patch the 2 travel DB-query helpers + REF + META stubs."""
        monkeypatch.setattr(
            inference_features,
            "_query_current_venue",
            lambda s, eid: curr_venue,
        )
        monkeypatch.setattr(
            inference_features,
            "_query_fighter_prior_venue",
            lambda s, fid, bd: prior_a if fid == 1 else prior_b,
        )
        # REF stub so the v2.2 dispatch doesn't blow up.
        monkeypatch.setattr(
            inference_features,
            "_query_ref_state",
            lambda s, r, d: ({}, {"finish": 0.5, "decision": 0.4, "no_action": 0.1}),
        )
        # META stubs (Plan 23-03) so MagicMock session is safe.
        monkeypatch.setattr(
            inference_features,
            "_query_fighter_prior_fight_date",
            lambda s, fid, bd: None,
        )
        monkeypatch.setattr(
            inference_features,
            "_query_elo_history",
            lambda s, fid, bd, limit=6: [],
        )
        monkeypatch.setattr(
            inference_features,
            "_query_division_state",
            lambda s, wc, bd: ({}, 0.0),
        )
        monkeypatch.setattr(
            inference_features,
            "_query_division_mean_reach",
            lambda s, wc: None,
        )
        monkeypatch.setattr(
            inference_features,
            "_query_fighter_division",
            lambda s, fid: None,
        )

    def test_inference_v22_travel_cols_present(self, monkeypatch) -> None:
        """Test 5: TRAVEL cols 75-80 populated (NON-NaN) when venue resolves."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS_V22

        session = _patch_session_with_snapshots(monkeypatch)
        curr_venue = {
            "lat": 34.0522,
            "lon": -118.2437,
            "timezone_iana": "America/Los_Angeles",
        }
        prior_a = {
            "lat": 40.7128,
            "lon": -74.0060,
            "timezone_iana": "America/New_York",
            "event_date": date(2026, 1, 1),
        }
        prior_b = {
            "lat": 34.0522,
            "lon": -118.2437,
            "timezone_iana": "America/Los_Angeles",
            "event_date": date(2026, 2, 1),
        }
        self._stub_travel_queries(monkeypatch, curr_venue, prior_a, prior_b)

        fa = _stub_fighter(1)
        fb = _stub_fighter(2)
        vec = inference_features.build(
            session,
            fa,
            fb,
            date(2026, 6, 14),
            feature_set="v2.2",
            event_id=42,
        )[0]

        # Fighter 1 (NYC→LA): travel ≈ 2451 mi.
        idx_red_dist = FEATURE_COLUMNS_V22.index("travel_distance_miles_red")
        idx_blue_dist = FEATURE_COLUMNS_V22.index("travel_distance_miles_blue")
        idx_diff_dist = FEATURE_COLUMNS_V22.index("travel_distance_miles_diff")
        idx_red_tz = FEATURE_COLUMNS_V22.index("tz_shift_red_signed")
        idx_blue_tz = FEATURE_COLUMNS_V22.index("tz_shift_blue_signed")
        idx_diff_tz = FEATURE_COLUMNS_V22.index("tz_shift_diff_signed")

        assert not np.isnan(vec[idx_red_dist])
        assert not np.isnan(vec[idx_blue_dist])
        # red (fighter 1, NYC→LA) ≈ 2451
        assert abs(vec[idx_red_dist] - 2451) < 15
        # blue (fighter 2, LA→LA) ≈ 0 (same venue)
        assert abs(vec[idx_blue_dist]) < 1.0
        # diff = red - blue
        assert vec[idx_diff_dist] == pytest.approx(
            vec[idx_red_dist] - vec[idx_blue_dist],
        )
        # diff_tz = red_tz - blue_tz
        assert vec[idx_diff_tz] == pytest.approx(
            vec[idx_red_tz] - vec[idx_blue_tz],
        )

    def test_inference_v22_travel_first_fight_sentinel(self, monkeypatch) -> None:
        """Test 6: debut fighter (no prior venue) → 0 sentinel per D-04."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS_V22

        session = _patch_session_with_snapshots(monkeypatch)
        curr_venue = {
            "lat": 34.0522,
            "lon": -118.2437,
            "timezone_iana": "America/Los_Angeles",
        }
        # Both fighters are debuting (no prior).
        self._stub_travel_queries(monkeypatch, curr_venue, None, None)

        fa = _stub_fighter(1)
        fb = _stub_fighter(2)
        vec = inference_features.build(
            session,
            fa,
            fb,
            date(2026, 6, 14),
            feature_set="v2.2",
            event_id=42,
        )[0]

        idx_red_dist = FEATURE_COLUMNS_V22.index("travel_distance_miles_red")
        idx_red_tz = FEATURE_COLUMNS_V22.index("tz_shift_red_signed")
        assert vec[idx_red_dist] == 0.0
        assert vec[idx_red_tz] == 0.0

    def test_inference_v22_travel_null_venue_propagates_nan(
        self,
        monkeypatch,
    ) -> None:
        """Test 7: Event.venue_id resolves to None → all 6 TRAVEL cols NaN.

        Graceful degradation per Pattern D — distinct from the debut
        sentinel 0 case.
        """
        session = _patch_session_with_snapshots(monkeypatch)
        # No current venue at all.
        self._stub_travel_queries(monkeypatch, None, None, None)

        fa = _stub_fighter(1)
        fb = _stub_fighter(2)
        vec = inference_features.build(
            session,
            fa,
            fb,
            date(2026, 6, 14),
            feature_set="v2.2",
            event_id=42,
        )[0]

        for idx in range(75, 81):
            assert np.isnan(vec[idx]), (
                f"col {idx} should NaN-degrade when curr_venue is None; got {vec[idx]}"
            )

    def test_inference_v22_travel_no_event_id(self, monkeypatch) -> None:
        """event_id=None → preserves NaN-init for TRAVEL cols (no DB lookup).

        Confirms _populate_travel returns early when event_id is None.
        """
        session = _patch_session_with_snapshots(monkeypatch)
        # Should never get called since event_id is None.
        called = {"curr": 0, "prior": 0}

        def _track_curr(s, eid):
            called["curr"] += 1
            return None

        def _track_prior(s, fid, bd):
            called["prior"] += 1
            return None

        monkeypatch.setattr(
            inference_features,
            "_query_current_venue",
            _track_curr,
        )
        monkeypatch.setattr(
            inference_features,
            "_query_fighter_prior_venue",
            _track_prior,
        )
        monkeypatch.setattr(
            inference_features,
            "_query_ref_state",
            lambda s, r, d: ({}, {"finish": 0.5, "decision": 0.4, "no_action": 0.1}),
        )
        # META stubs (Plan 23-03) so MagicMock session is safe.
        monkeypatch.setattr(
            inference_features,
            "_query_fighter_prior_fight_date",
            lambda s, fid, bd: None,
        )
        monkeypatch.setattr(
            inference_features,
            "_query_elo_history",
            lambda s, fid, bd, limit=6: [],
        )
        monkeypatch.setattr(
            inference_features,
            "_query_division_state",
            lambda s, wc, bd: ({}, 0.0),
        )
        monkeypatch.setattr(
            inference_features,
            "_query_division_mean_reach",
            lambda s, wc: None,
        )
        monkeypatch.setattr(
            inference_features,
            "_query_fighter_division",
            lambda s, fid: None,
        )

        fa = _stub_fighter(1)
        fb = _stub_fighter(2)
        vec = inference_features.build(
            session,
            fa,
            fb,
            date(2026, 6, 14),
            feature_set="v2.2",
            event_id=None,
        )[0]

        # All 6 TRAVEL cols NaN (init preserved).
        for idx in range(75, 81):
            assert np.isnan(vec[idx]), f"col {idx} should be NaN"
        # _populate_travel returned EARLY → no DB lookups attempted.
        assert called["curr"] == 0, "Should not query current venue if event_id is None"
        assert called["prior"] == 0, "Should not query prior venue if event_id is None"


# ── TestAgeBugRegression (Phase 23 Plan 23-03 — Pitfall #5 fix) ─────────────


class TestAgeBugRegression:
    """Phase 23 Plan 23-03 — Pitfall #5 age-bug regression guard.

    The pre-fix behavior used `date.today()` to compute age_diff in
    `_populate_physical`, causing inference to diverge from training for any
    backtest at a historical event_date. The fix uses the `event_date`
    parameter that `build()` already accepts.

    This test asserts deterministic age_diff for a fixed (DoB, event_date)
    pair — its expected value is INDEPENDENT of the date the test runs on,
    so any return of the bug fails the test immediately.
    """

    def test_inference_age_uses_event_date_not_today(self, monkeypatch) -> None:
        """Age computed from event_date — not date.today()."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS

        session = _patch_session_with_snapshots(monkeypatch)

        # Fighter A: DoB 1990-01-01 (age at 2020-01-01 = exactly 30.0 yrs)
        # Fighter B: DoB 1995-01-01 (age at 2020-01-01 = exactly 25.0 yrs)
        # Expected age_diff = 30 - 25 = 5.0 (computed via /365.25 → ~4.9966)
        fa = _stub_fighter(1, date_of_birth=date(1990, 1, 1))
        fb = _stub_fighter(2, date_of_birth=date(1995, 1, 1))

        vec = inference_features.build(
            session,
            fa,
            fb,
            date(2020, 1, 1),
        )[0]

        idx_age = FEATURE_COLUMNS.index("age_diff")
        age_diff = vec[idx_age]

        # Compute expected from same arithmetic the code MUST use.
        age_a = (date(2020, 1, 1) - date(1990, 1, 1)).days / 365.25
        age_b = (date(2020, 1, 1) - date(1995, 1, 1)).days / 365.25
        expected = age_a - age_b
        assert age_diff == pytest.approx(expected, abs=1e-9), (
            f"Pitfall #5 regression: age_diff at event_date=2020-01-01 "
            f"got {age_diff}, expected {expected} (event-date-based). "
            f"If this assertion fails because age_diff matches a "
            f"`date.today()`-based value, the bug has been re-introduced."
        )
        # Also assert the value is in the right ballpark for the event_date
        # — independent of whatever 'today' is.
        assert abs(age_diff - 5.0) < 0.01, (
            f"Pitfall #5 regression: age_diff={age_diff} should be ~5.0 "
            f"(30 - 25); a value ~30+ years (today() - 1990-01-01 minus "
            f"today() - 1995-01-01 = ~5 but at adult ages) suggests the "
            f"divergence is hidden — re-check whether event_date is honored."
        )

    def test_inference_no_date_today_in_age_block(self) -> None:
        """`_populate_physical` must NOT call date.today() for age math.

        Greps the inference_features.py source for the buggy pattern. Belt-
        and-suspenders complement to the value-based regression test above.
        """
        import subprocess

        # The exact buggy pattern: `today = date.today()` followed by age math.
        result = subprocess.run(
            ["grep", "-n", "today = date.today()", "src/ufc_prediction/ml/inference_features.py"],
            capture_output=True,
            text=True,
        )
        # grep returncode 1 = no matches (desired state post-fix).
        assert result.returncode == 1, (
            f"Pitfall #5 regression: buggy `today = date.today()` line "
            f"present in inference_features.py:\n{result.stdout}"
        )
