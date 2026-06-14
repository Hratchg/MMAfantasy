"""Phase 23 Plan 23-04 — end-to-end integration smoke test (fixture-only).

Verifies the full v2.2 training-path pipeline produces a 90-col matrix with
APPEND-ONLY discipline and AUDIT-01 SHA invariant preserved. Uses pure dict
fixtures (no SQLAlchemy session, no ORM, no DB fixture) per planner-revision
B-3 Option A — the DB-backed train/inference parity assertions live at the
unit-test level in
``tests/unit/ml/test_features_v22.py::TestTrainInferenceParity*`` (Plans
23-01/02/03).

Phase 23 final column count is **90** (72 NO_NET baseline + 3 REF + 6 TRAVEL
+ 9 META), NOT 91 — ``layoff_days_diff`` was dropped per RESEARCH Q4
RESOLVED + CONTEXT D-07 (semantic duplicate of
``FEATURE_COLUMNS_NO_NET[61]`` = ``days_since_last_fight_diff``).
``age_diff`` is preserved at ``FEATURE_COLUMNS_NO_NET[26]`` per APPEND-ONLY
discipline (Plan 23-01 deferred-items log).
"""

from __future__ import annotations

import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from ufc_prediction.ml.config import (
    FEATURE_COLUMNS_NO_NET,
    FEATURE_COLUMNS_V22,
)
from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler


# ─── Repo root + audit script path (resolved at import time) ────────────────

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AUDIT_SCRIPT = _REPO_ROOT / "scripts" / "audit_xgb_v2_sha.sh"


# ─── Tiny self-contained fixture (no DB session; no ORM) ────────────────────


@pytest.fixture()
def tiny_v22_corpus() -> dict:
    """5 fights / 2 referees / 3 venues / 5 fighters — pure dict fixture.

    Returned dict is destructure-ready as ``**tiny_v22_corpus`` into
    ``FeatureMatrixAssembler().assemble()``. All required keys for the
    v2.2 dispatch (event_id, referee_id, method, venue_lat/lon/timezone_iana)
    are inlined on each fight_records dict; the assembler's v2.2 pre-pass
    will auto-build event_referees / ref_history / event_venues /
    fighter_prior_venues / fighter_birth_dates / fighter_reaches /
    elo_histories / fighter_prior_fight_dates / division_history /
    global_finish_rate / division_mean_reaches from those inline fields.

    Layout:
      - F1, F2 (Lightweight; F1=NYC native, F2 travels NYC→LA→Vegas)
      - F3, F4 (Welterweight; F3 partial physicals, F4 missing physicals)
      - F5 (Lightweight; debuts at fight 104 → debut sentinel = 0)
      - Referees: R301 (3 fights: 101, 103, 105) + R302 (2 fights: 102, 104)
      - Venues: V_NYC (101, 104), V_LA (102, 103), V_LV (105)
      - Methods give finish + decision + no_action mix for non-trivial REF rates
      - >720-day gap between fights 102 (2021-06-20) and 103 (2022-11-05)
        for fighter 1 to exercise per-fighter layoff clip
    """
    fighter_physicals: dict[int, dict] = {
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
            "date_of_birth": date(1995, 11, 20),  # debuts at fight 104
        },
    }

    # Venue cookbook: (lat, lon, tz_iana)
    V_NYC = (40.7128, -74.0060, "America/New_York")
    V_LA = (34.0522, -118.2437, "America/Los_Angeles")
    V_LV = (36.1699, -115.1398, "America/Los_Angeles")

    # 5 fights chronologically — deterministic dates, refs, venues, methods.
    # Note: F5 first fight is 104 (debuts → travel sentinel = 0).
    # F1 layoff between 102 (not-applicable: F1 not in 102) and 103 = (2022-11-05
    # - 2020-01-15) = 1025 days → 720-clip exercised at meta layer.
    fight_records: list[dict] = [
        {
            "fight_id": 101,
            "event_id": 201,
            "event_date": date(2020, 1, 15),
            "fighter_a_id": 1,
            "fighter_b_id": 2,
            "winner_id": 1,
            "weight_class": "Lightweight",
            "referee_id": 301,
            "method": "KO/TKO",
            "venue_lat": V_NYC[0],
            "venue_lon": V_NYC[1],
            "venue_timezone_iana": V_NYC[2],
        },
        {
            "fight_id": 102,
            "event_id": 202,
            "event_date": date(2021, 6, 20),
            "fighter_a_id": 3,
            "fighter_b_id": 4,
            "winner_id": 4,
            "weight_class": "Welterweight",
            "referee_id": 302,
            "method": "Decision - Unanimous",
            "venue_lat": V_LA[0],
            "venue_lon": V_LA[1],
            "venue_timezone_iana": V_LA[2],
        },
        {
            "fight_id": 103,
            "event_id": 203,
            "event_date": date(2022, 11, 5),
            "fighter_a_id": 1,
            "fighter_b_id": 2,
            "winner_id": 2,
            "weight_class": "Lightweight",
            "referee_id": 301,
            "method": "Submission",
            "venue_lat": V_LA[0],
            "venue_lon": V_LA[1],
            "venue_timezone_iana": V_LA[2],
        },
        {
            "fight_id": 104,
            "event_id": 204,
            "event_date": date(2023, 3, 10),
            "fighter_a_id": 2,
            "fighter_b_id": 5,  # F5 debut
            "winner_id": 2,
            "weight_class": "Lightweight",
            "referee_id": 302,
            "method": "Decision - Split",
            "venue_lat": V_NYC[0],
            "venue_lon": V_NYC[1],
            "venue_timezone_iana": V_NYC[2],
        },
        {
            "fight_id": 105,
            "event_id": 205,
            "event_date": date(2024, 1, 20),
            "fighter_a_id": 1,
            "fighter_b_id": 5,
            "winner_id": 1,
            "weight_class": "Lightweight",
            "referee_id": 301,
            "method": "Decision - Unanimous",
            "venue_lat": V_LV[0],
            "venue_lon": V_LV[1],
            "venue_timezone_iana": V_LV[2],
        },
    ]

    # Elo features: (fighter_id, fight_id) -> {elo_overall, elo_striking, elo_grappling}
    # Deterministic monotone progression so velocity diffs are well-defined.
    elo_features: dict[tuple[int, int], dict[str, float]] = {
        (1, 101): {"elo_overall": 1520.0, "elo_striking": 1530.0, "elo_grappling": 1510.0},
        (2, 101): {"elo_overall": 1480.0, "elo_striking": 1470.0, "elo_grappling": 1490.0},
        (3, 102): {"elo_overall": 1500.0, "elo_striking": 1500.0, "elo_grappling": 1500.0},
        (4, 102): {"elo_overall": 1510.0, "elo_striking": 1505.0, "elo_grappling": 1515.0},
        (1, 103): {"elo_overall": 1540.0, "elo_striking": 1550.0, "elo_grappling": 1520.0},
        (2, 103): {"elo_overall": 1495.0, "elo_striking": 1485.0, "elo_grappling": 1505.0},
        (2, 104): {"elo_overall": 1505.0, "elo_striking": 1495.0, "elo_grappling": 1515.0},
        (5, 104): {"elo_overall": 1500.0, "elo_striking": 1500.0, "elo_grappling": 1500.0},
        (1, 105): {"elo_overall": 1555.0, "elo_striking": 1565.0, "elo_grappling": 1535.0},
        (5, 105): {"elo_overall": 1520.0, "elo_striking": 1515.0, "elo_grappling": 1525.0},
    }

    # The 20-numeric-feature canonical schema (computed_features values).
    # Same shape used by tests/ml/conftest.py — keeps the v2.2 row width valid.
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
    computed_features: dict[tuple[int, int], dict[str, float | None]] = {
        (1, 101): dict(base_features),
        (2, 101): dict(alt_features),
        (3, 102): dict(base_features),
        (4, 102): dict(alt_features),
        (1, 103): dict(base_features),
        (2, 103): dict(alt_features),
        (2, 104): dict(alt_features),
        (5, 104): dict(base_features),
        (1, 105): dict(base_features),
        (5, 105): dict(alt_features),
    }

    division_medians: dict[str, dict[str, float]] = {
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

    return dict(
        fight_records=fight_records,
        elo_features=elo_features,
        computed_features=computed_features,
        fighter_physicals=fighter_physicals,
        division_medians=division_medians,
    )


# ─── Tests ──────────────────────────────────────────────────────────────────


class TestPhase23FullPipelineV22:
    """End-to-end Phase 23 phase-close integration smoke test."""

    # ── FEATURE_COLUMNS_V22 invariants (config-layer) ───────────────────

    def test_feature_columns_v22_length_90(self) -> None:
        """Q4 RESOLVED + D-07 + Plan 23-01 deferred-items:
        layoff_days_diff DROPPED → final length is 90 (NOT 91, NOT 92).
        72 NO_NET + 3 REF + 6 TRAVEL + 9 META = 90."""
        assert len(FEATURE_COLUMNS_V22) == 90, (
            f"Expected 90 cols (72 NO_NET + 3 REF + 6 TRAVEL + 9 META post-Q4); "
            f"got {len(FEATURE_COLUMNS_V22)}"
        )

    def test_feature_columns_v22_no_layoff_days_diff(self) -> None:
        """Q4 RESOLVED + D-07: layoff_days_diff must NOT be in FEATURE_COLUMNS_V22.

        days_since_last_fight_diff at FEATURE_COLUMNS_NO_NET[61] is the
        canonical differential. Adding layoff_days_diff would be a D-07
        dedup violation (semantic duplicate).
        """
        assert "layoff_days_diff" not in FEATURE_COLUMNS_V22

    def test_feature_columns_v22_append_only(self) -> None:
        """APPEND-ONLY (D-08): NO_NET preserved byte-identical at indices 0-71.

        Mutating FEATURE_COLUMNS_NO_NET would invalidate the xgb_v2.joblib
        baseline (AUDIT-01 chain) — APPEND-ONLY discipline forbids it.
        """
        assert FEATURE_COLUMNS_V22[:72] == FEATURE_COLUMNS_NO_NET

    def test_feature_columns_v22_dedup_at_integration(self) -> None:
        """D-07 dedup: zero overlap between V22 tail (19 new cols) and NO_NET."""
        overlap = set(FEATURE_COLUMNS_V22[72:]) & set(FEATURE_COLUMNS_NO_NET)
        assert overlap == set(), (
            f"D-07 dedup violation — V22 tail overlaps NO_NET: {overlap}"
        )

    def test_feature_columns_v22_index_spot_checks(self) -> None:
        """Known column positions across REF / TRAVEL / META blocks.

        Verifies the static FEATURE_COLUMNS_V22 layout matches the contract
        documented in 23-01/02/03 SUMMARYs. Catches accidental re-orderings
        that would silently break xgb_v2 ↔ v2.2 train/inference parity.
        """
        # REF block (72-74)
        assert FEATURE_COLUMNS_V22[72] == "ref_finish_rate_shrunk"
        assert FEATURE_COLUMNS_V22[73] == "ref_decision_rate_shrunk"
        assert FEATURE_COLUMNS_V22[74] == "ref_no_action_rate_shrunk"
        # TRAVEL block (75-80)
        assert FEATURE_COLUMNS_V22[75] == "travel_distance_miles_red"
        assert FEATURE_COLUMNS_V22[76] == "travel_distance_miles_blue"
        assert FEATURE_COLUMNS_V22[77] == "travel_distance_miles_diff"
        assert FEATURE_COLUMNS_V22[78] == "tz_shift_red_signed"
        assert FEATURE_COLUMNS_V22[79] == "tz_shift_blue_signed"
        assert FEATURE_COLUMNS_V22[80] == "tz_shift_diff_signed"
        # META block (81-89; 9 cols — NO layoff_days_diff per Q4)
        assert FEATURE_COLUMNS_V22[81] == "layoff_days_red"
        assert FEATURE_COLUMNS_V22[82] == "layoff_days_blue"
        assert FEATURE_COLUMNS_V22[83] == "age_at_fight_red"
        assert FEATURE_COLUMNS_V22[84] == "age_at_fight_blue"
        assert FEATURE_COLUMNS_V22[85] == "elo_overall_velocity_diff"
        assert FEATURE_COLUMNS_V22[86] == "elo_striking_velocity_diff"
        assert FEATURE_COLUMNS_V22[87] == "elo_grappling_velocity_diff"
        assert FEATURE_COLUMNS_V22[88] == "division_finish_rate_shrunk"
        assert FEATURE_COLUMNS_V22[89] == "reach_diff_normalized"

    def test_feature_columns_v22_preserves_age_diff_at_no_net_index(self) -> None:
        """age_diff is preserved at FEATURE_COLUMNS_NO_NET[26] per APPEND-ONLY.

        The v2.2 design adds per-fighter age_at_fight_red/blue (indices
        83-84) as NEW META cols WITHOUT removing the existing age_diff at
        index 26 of the NO_NET baseline. APPEND-ONLY discipline forbids
        mutating NO_NET.
        """
        assert "age_diff" in FEATURE_COLUMNS_NO_NET
        assert "age_diff" in FEATURE_COLUMNS_V22
        # age_diff is in NO_NET (preserved), NOT in the 19 new V22-tail cols
        assert "age_diff" not in FEATURE_COLUMNS_V22[72:]

    # ── Training-path assemble() shape contracts ────────────────────────

    def test_assemble_v22_produces_90_cols(self, tiny_v22_corpus: dict) -> None:
        """Full v2.2 dispatch on the tiny corpus → (5, 90) feature matrix.

        Exercises every v2.2 pre-pass builder (REF + TRAVEL + META) end-to-end
        without any DB session — proof that the training path can compute the
        full 90-col v2.2 vector from pure-dict inputs.
        """
        assembler = FeatureMatrixAssembler()
        X, y, fight_dates = assembler.assemble(
            **tiny_v22_corpus, feature_set="v2.2",
        )
        assert X.shape == (5, 90), f"Expected (5, 90); got {X.shape}"
        assert X.shape[1] == len(FEATURE_COLUMNS_V22)
        assert y.shape == (5,)
        assert fight_dates.shape == (5,)
        # X is float64 — np.isnan is safe; verify no DTYPE drift.
        assert X.dtype == np.float64

    def test_assemble_v21_backcompat_72_cols(self, tiny_v22_corpus: dict) -> None:
        """Back-compat: feature_set='v2.1-no-net' produces 72-col output."""
        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(**tiny_v22_corpus, feature_set="v2.1-no-net")
        assert X.shape == (5, 72)

    def test_assemble_include_net_false_backcompat(self, tiny_v22_corpus: dict) -> None:
        """Back-compat: include_net=False maps to feature_set='v2.1-no-net' → 72 cols.

        Phase 18 shim preservation — xgb_v2 callers using include_net=False
        must continue to receive the 72-col baseline.
        """
        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(**tiny_v22_corpus, include_net=False)
        assert X.shape == (5, 72)

    def test_assemble_v22_ref_block_populated_not_nan(
        self, tiny_v22_corpus: dict,
    ) -> None:
        """REF block (cols 72-74) populated from referee_id + method fields.

        Every fight in the fixture has a non-None referee_id and method →
        REF helpers should produce finite values (debutant referees collapse
        to global rate via Bayesian fallback, still finite).
        """
        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(**tiny_v22_corpus, feature_set="v2.2")
        ref_slice = X[:, 72:75]
        assert not np.any(np.isnan(ref_slice)), (
            f"REF block must be non-NaN when referee_id+method populated; "
            f"got {ref_slice}"
        )

    def test_assemble_v22_travel_block_populated_not_nan(
        self, tiny_v22_corpus: dict,
    ) -> None:
        """TRAVEL block (cols 75-80) populated when venue_lat/lon/tz_iana present.

        Debut fighters receive 0 sentinel (D-04); subsequent fighters get
        haversine + tz_shift. Either way: non-NaN.
        """
        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(**tiny_v22_corpus, feature_set="v2.2")
        travel_slice = X[:, 75:81]
        assert not np.any(np.isnan(travel_slice)), (
            f"TRAVEL block must be non-NaN when venues populated; "
            f"got {travel_slice}"
        )

    def test_assemble_v22_debut_fighter_travel_sentinel_zero(
        self, tiny_v22_corpus: dict,
    ) -> None:
        """D-04: debut fighter (no prior venue) → travel + tz_shift = 0 sentinel.

        At fight 101 (chronologically earliest), both F1 and F2 are debuting
        in this corpus → both red+blue travel_distance + diff and tz_shift +
        diff must be exactly 0 (NOT NaN, NOT some haversine residual).
        """
        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(**tiny_v22_corpus, feature_set="v2.2")
        # fight 101 is index 0 (already chronologically first in fixture)
        idx_101 = next(
            i for i, f in enumerate(tiny_v22_corpus["fight_records"])
            if f["fight_id"] == 101
        )
        assert X[idx_101, 75] == 0.0, "travel_distance_miles_red debut sentinel"
        assert X[idx_101, 76] == 0.0, "travel_distance_miles_blue debut sentinel"
        assert X[idx_101, 77] == 0.0, "travel_distance_miles_diff debut sentinel"
        assert X[idx_101, 78] == 0.0, "tz_shift_red_signed debut sentinel"
        assert X[idx_101, 79] == 0.0, "tz_shift_blue_signed debut sentinel"
        assert X[idx_101, 80] == 0.0, "tz_shift_diff_signed debut sentinel"

    def test_assemble_v22_age_uses_event_date_not_today(
        self, tiny_v22_corpus: dict,
    ) -> None:
        """Pitfall #5 regression: age_at_fight uses event_date (training path).

        Fight 101 has F1 (DoB=1990-01-01, age≈30) and F2 (DoB=1988-06-15,
        age≈31.58) at event_date 2020-01-15. The assembler applies a
        deterministic A/B hash swap per fight_id (see feature_matrix.py:794
        ~ ``hashlib.md5(str(fight_id))``), so the {red, blue} order at any
        given row may be either (F1, F2) or (F2, F1). We verify that the
        emitted age pair is the SET {30, 31.58} regardless of which side
        each fighter lands on. If Pitfall #5 had regressed (date.today()
        used instead of event_date), today's date in 2026 would yield
        ages ~36 + ~37.5 — neither value would match.
        """
        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(**tiny_v22_corpus, feature_set="v2.2")
        idx_101 = next(
            i for i, f in enumerate(tiny_v22_corpus["fight_records"])
            if f["fight_id"] == 101
        )
        age_red = X[idx_101, 83]
        age_blue = X[idx_101, 84]
        # The deterministic SET — invariant under any A/B swap and pinned
        # to event_date, NOT to date.today(). If Pitfall #5 regresses, BOTH
        # values will be ~6 years larger (today's 2026 date).
        emitted_pair = sorted([age_red, age_blue])
        # F1 born 1990-01-01 → at 2020-01-15 = 30 + 14 days ≈ 30.04
        # F2 born 1988-06-15 → at 2020-01-15 = 31 + 214 days ≈ 31.59
        assert 29.9 <= emitted_pair[0] <= 30.2, (
            f"younger age (~30.04) expected; got {emitted_pair[0]} "
            f"— Pitfall #5 may have regressed (date.today()?)"
        )
        assert 31.4 <= emitted_pair[1] <= 31.8, (
            f"older age (~31.59) expected; got {emitted_pair[1]} "
            f"— Pitfall #5 may have regressed (date.today()?)"
        )

    def test_assemble_v22_layoff_clip_at_720_days(
        self, tiny_v22_corpus: dict,
    ) -> None:
        """D-06: per-fighter layoff clipped at 720 days.

        Fight 103 (2022-11-05) is F1's 2nd UFC bout (1st was fight 101 on
        2020-01-15, gap = 1025 days) and F2's 2nd bout (1st = fight 101
        same date, also 1025 days). Both fighters have raw gap >720 →
        BOTH layoff_days_red and layoff_days_blue at fight 103 must be
        clipped to ≤720, and NEITHER must equal the unclipped 1025.
        """
        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(**tiny_v22_corpus, feature_set="v2.2")
        idx_103 = next(
            i for i, f in enumerate(tiny_v22_corpus["fight_records"])
            if f["fight_id"] == 103
        )
        layoff_red = X[idx_103, 81]
        layoff_blue = X[idx_103, 82]
        assert layoff_red <= 720.0, (
            f"layoff_days_red must be clipped at 720; got {layoff_red}"
        )
        assert layoff_blue <= 720.0, (
            f"layoff_days_blue must be clipped at 720; got {layoff_blue}"
        )
        # And the true raw gap (1025 days) must NOT pass through unclipped
        assert layoff_red != 1025.0
        assert layoff_blue != 1025.0

    # ── AUDIT-01 SHA invariant (Phase 23 end-of-chain checkpoint) ──────

    def test_audit_01_sha_invariant_at_phase_end(self) -> None:
        """AUDIT-01 chain: xgb_v2.joblib SHA-256 byte-identical to baseline.

        Phase 23 added 19 new feature cols + computation paths WITHOUT
        retraining xgb_v2 — the baseline model is consumed via
        feature_set='v2.1-no-net'. SHA invariance is the AUDIT-01 guarantee.
        """
        result = subprocess.run(
            ["bash", str(_AUDIT_SCRIPT)],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"AUDIT-01 SHA drift at phase end: rc={result.returncode}; "
            f"stdout={result.stdout!r}; stderr={result.stderr!r}"
        )

    def test_audit_01_baseline_sha_present(self) -> None:
        """The AUDIT-01 baseline SHA file exists with the canonical hash."""
        baseline = _REPO_ROOT / ".planning" / "AUDIT-01-BASELINE-SHA.txt"
        assert baseline.exists(), "AUDIT-01-BASELINE-SHA.txt missing"
        content = baseline.read_text().strip()
        assert content == (
            "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"
        ), f"Baseline SHA changed: {content}"
