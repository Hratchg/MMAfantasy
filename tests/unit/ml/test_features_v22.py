"""Tests for Phase 23 v2.2 feature helpers (REF + scaffolding).

Phase 23 Plan 23-01 RED-then-GREEN tests for:
    - Beta-binomial EB shrinkage helper (D-02)
    - classify_outcome method-string categorization (Q2 RESOLVED)
    - compute_ref_rates_shrunk pure-function REF compute (D-03)
    - FEATURE_COLUMNS_V22 APPEND-ONLY shape (D-08)
    - get_feature_columns(feature_set=) dispatch (D-09)

Per Pitfall #12: this module is the single source of truth for v2.2 math;
both feature_matrix.py and inference_features.py call into it for parity.
"""

from __future__ import annotations

from datetime import date

import pytest

# ── TestBetaBinomialShrink ──────────────────────────────────────────────────


class TestBetaBinomialShrink:
    def test_beta_binomial_zero_collapse(self) -> None:
        """Test 1: n=0 collapses to global_rate (Bayesian fallback per D-02)."""
        from ufc_prediction.ml.features_v22 import beta_binomial_shrink

        # (0 + 50*0.4) / (0 + 50) = 20 / 50 = 0.4
        assert beta_binomial_shrink(0, 0, 0.4, 50.0) == pytest.approx(0.4)

    def test_beta_binomial_full_evidence(self) -> None:
        """Test 2: large-n posterior approaches the empirical rate."""
        from ufc_prediction.ml.features_v22 import beta_binomial_shrink

        # (10 + 50*0.4) / (10 + 50) = 30 / 60 = 0.5
        assert beta_binomial_shrink(10, 10, 0.4, 50.0) == pytest.approx(0.5)

    def test_beta_binomial_bounded(self) -> None:
        """Test 3: output ∈ [0, 1] for any (successes ≤ n) and global ∈ [0, 1]."""
        from ufc_prediction.ml.features_v22 import beta_binomial_shrink

        for successes, n in [(0, 0), (0, 10), (5, 10), (10, 10), (3, 100), (100, 100)]:
            for global_rate in [0.0, 0.1, 0.5, 0.9, 1.0]:
                v = beta_binomial_shrink(successes, n, global_rate, 50.0)
                assert 0.0 <= v <= 1.0, (
                    f"Out-of-bounds: successes={successes} n={n} g={global_rate} → {v}"
                )


# ── TestClassifyOutcome ─────────────────────────────────────────────────────


class TestClassifyOutcome:
    def test_classify_outcome_finish(self) -> None:
        """Test 4: KO/TKO/Submission/SUB → finish."""
        from ufc_prediction.ml.features_v22 import classify_outcome

        assert classify_outcome("KO/TKO") == "finish"
        assert classify_outcome("Submission") == "finish"
        assert classify_outcome("SUB") == "finish"
        assert classify_outcome("TKO") == "finish"

    def test_classify_outcome_decision(self) -> None:
        """Test 5: Decision variants → decision."""
        from ufc_prediction.ml.features_v22 import classify_outcome

        assert classify_outcome("Decision - Unanimous") == "decision"
        assert classify_outcome("Decision - Split") == "decision"
        assert classify_outcome("Decision - Majority") == "decision"
        assert classify_outcome("Decision") == "decision"

    def test_classify_outcome_no_action(self) -> None:
        """Test 6: NC / DQ / None / unknown → no_action (defensive default)."""
        from ufc_prediction.ml.features_v22 import classify_outcome

        assert classify_outcome("No Contest") == "no_action"
        assert classify_outcome(None) == "no_action"
        assert classify_outcome("UNKNOWN_METHOD") == "no_action"
        assert classify_outcome("DQ") == "no_action"
        assert classify_outcome("Disqualification") == "no_action"
        assert classify_outcome("Overturned") == "no_action"


# ── TestComputeRefRatesShrunk ───────────────────────────────────────────────


class TestComputeRefRatesShrunk:
    def test_compute_ref_rates_unknown_referee(self) -> None:
        """Test 7: referee_id=None or unknown → all 3 collapse to global_rates."""
        from ufc_prediction.ml.features_v22 import compute_ref_rates_shrunk

        global_rates = {"finish": 0.5, "decision": 0.4, "no_action": 0.1}

        # Case A: referee_id is None
        out = compute_ref_rates_shrunk(None, date(2026, 1, 1), {}, global_rates)
        assert out["ref_finish_rate_shrunk"] == pytest.approx(0.5)
        assert out["ref_decision_rate_shrunk"] == pytest.approx(0.4)
        assert out["ref_no_action_rate_shrunk"] == pytest.approx(0.1)

        # Case B: referee_id present but no history entry → same fallback
        out2 = compute_ref_rates_shrunk(99, date(2026, 1, 1), {}, global_rates)
        assert out2["ref_finish_rate_shrunk"] == pytest.approx(0.5)
        assert out2["ref_decision_rate_shrunk"] == pytest.approx(0.4)
        assert out2["ref_no_action_rate_shrunk"] == pytest.approx(0.1)

    def test_compute_ref_rates_no_temporal_leakage(self) -> None:
        """Test 8: strict event_date < as_of_date — Pitfall #4 leakage guard."""
        from ufc_prediction.ml.features_v22 import (
            beta_binomial_shrink,
            compute_ref_rates_shrunk,
        )

        ref_history = {
            7: [
                {"event_date": date(2025, 1, 1), "method": "KO/TKO"},
                {"event_date": date(2025, 6, 1), "method": "Decision - Unanimous"},
            ]
        }
        global_rates = {"finish": 0.5, "decision": 0.4, "no_action": 0.1}

        out = compute_ref_rates_shrunk(
            7,
            date(2025, 3, 1),
            ref_history,
            global_rates,
            k_shrink=50.0,
        )
        # Only 2025-01-01 (finish) fight visible at as_of=2025-03-01.
        # n=1, finishes=1, decisions=0, no_action=0.
        exp_finish = beta_binomial_shrink(1, 1, 0.5, 50.0)
        exp_decision = beta_binomial_shrink(0, 1, 0.4, 50.0)
        exp_no_action = beta_binomial_shrink(0, 1, 0.1, 50.0)
        assert out["ref_finish_rate_shrunk"] == pytest.approx(exp_finish)
        assert out["ref_decision_rate_shrunk"] == pytest.approx(exp_decision)
        assert out["ref_no_action_rate_shrunk"] == pytest.approx(exp_no_action)

        # Reverse check: at as_of=2026-01-01 BOTH fights visible.
        out2 = compute_ref_rates_shrunk(
            7,
            date(2026, 1, 1),
            ref_history,
            global_rates,
            k_shrink=50.0,
        )
        exp_finish2 = beta_binomial_shrink(1, 2, 0.5, 50.0)
        exp_decision2 = beta_binomial_shrink(1, 2, 0.4, 50.0)
        exp_no_action2 = beta_binomial_shrink(0, 2, 0.1, 50.0)
        assert out2["ref_finish_rate_shrunk"] == pytest.approx(exp_finish2)
        assert out2["ref_decision_rate_shrunk"] == pytest.approx(exp_decision2)
        assert out2["ref_no_action_rate_shrunk"] == pytest.approx(exp_no_action2)


# ── TestTrainInferenceParity (LIVE-03 D-10) ─────────────────────────────────


class TestTrainInferenceParity:
    """Phase 23 D-10 — REF cols byte-identical between training and inference
    paths.

    This is THE parity guard: feature_matrix.assemble(...feature_set='v2.2')
    and inference_features.build(...feature_set='v2.2') must emit IDENTICAL
    values for the 3 REF cols (indices 72-74) given the same fixture data.

    The training path computes ref_history + global_rates from the
    fight_records corpus in a pre-pass; the inference path queries them from
    the DB. Here we wire the inference path to read from a stub of the same
    in-memory corpus, then assert the REF values match the training path
    bit-for-bit.
    """

    def test_ref_cols_parity_between_train_and_inference(
        self,
        monkeypatch,
    ) -> None:
        """Test 8: REF cols at indices 72-74 are byte-identical across paths."""
        from datetime import date as _date
        from unittest.mock import MagicMock

        import numpy as np
        import pytest as _pytest

        from ufc_prediction.ml import inference_features
        from ufc_prediction.ml.feature_matrix import (
            FeatureMatrixAssembler,
            _build_referee_lookup,
            _compute_ref_global_rates,
        )

        # ── Shared fixture: 3 fights, 2 referees, 3 methods ────────────────
        # Same shape the training path expects; we replay the same data
        # through the inference path's _query_ref_state stub.
        fight_records = [
            {
                "fight_id": 201,
                "event_id": 201,
                "event_date": _date(2024, 1, 1),
                "fighter_a_id": 1,
                "fighter_b_id": 2,
                "winner_id": 1,
                "weight_class": "Lightweight",
                "referee_id": 1,
                "method": "KO/TKO",
            },
            {
                "fight_id": 202,
                "event_id": 202,
                "event_date": _date(2024, 6, 1),
                "fighter_a_id": 3,
                "fighter_b_id": 4,
                "winner_id": 3,
                "weight_class": "Welterweight",
                "referee_id": 2,
                "method": "Decision - Unanimous",
            },
            {
                "fight_id": 203,
                "event_id": 203,
                "event_date": _date(2025, 3, 1),
                "fighter_a_id": 1,
                "fighter_b_id": 3,
                "winner_id": 1,
                "weight_class": "Lightweight",
                "referee_id": 1,
                "method": "Submission",
            },
        ]

        # Minimal Elo / computed / physical / division fixtures.
        elo_default = {
            "elo_overall": 1500.0,
            "elo_striking": 1500.0,
            "elo_grappling": 1500.0,
        }
        elo_features = {
            (fid, fight["fight_id"]): dict(elo_default)
            for fight in fight_records
            for fid in (fight["fighter_a_id"], fight["fighter_b_id"])
        }
        computed_features: dict = {}  # NaN-propagating, fine for REF parity test
        fighter_physicals = {
            fid: {
                "height_inches": 70.0,
                "reach_inches": 72.0,
                "leg_reach_inches": 40.0,
                "stance": "Orthodox",
                "date_of_birth": _date(1990, 1, 1),
            }
            for fid in (1, 2, 3, 4)
        }
        division_medians = {
            "Lightweight": {
                "height_inches": 70.0,
                "reach_inches": 72.0,
                "leg_reach_inches": 40.0,
            },
            "Welterweight": {
                "height_inches": 70.0,
                "reach_inches": 72.0,
                "leg_reach_inches": 40.0,
            },
        }

        # ── Compute shared ref_history + global_rates ───────────────────────
        # We inject the SAME inputs into BOTH paths so the parity gate is a
        # pure compute-helper-equivalence check (Pitfall #12 single source
        # of truth). The training path natively builds these from the
        # corpus via _compute_ref_global_rates + _build_referee_lookup; the
        # inference path queries them from the DB via _query_ref_state.
        # When fed the SAME inputs they MUST produce the same REF rates.
        sorted_records = sorted(fight_records, key=lambda f: f["event_date"])
        shared_global_rates = _compute_ref_global_rates(sorted_records)
        event_refs, shared_ref_history = _build_referee_lookup(sorted_records)

        # ── Training path: pass shared inputs explicitly ───────────────────
        # By passing ref_global_rates and ref_history kwargs we bypass the
        # in-method pre-pass and force the assembler to use the same dicts
        # the inference path will see.
        assembler = FeatureMatrixAssembler()
        X_train, _, _ = assembler.assemble(
            fight_records,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
            feature_set="v2.2",
            event_referees=event_refs,
            ref_history=shared_ref_history,
            ref_global_rates=shared_global_rates,
        )

        # ── Inference path stub: return the SAME shared inputs ─────────────
        def _stub_query_ref_state(session, referee_id, the_event_date):
            # Return the FULL shared_ref_history; compute_ref_rates_shrunk
            # applies the strict event_date < as_of_date cutoff internally,
            # so both paths use the same temporal filter.
            return shared_ref_history, shared_global_rates

        monkeypatch.setattr(
            inference_features,
            "_get_latest_elo",
            lambda s, fid, et: 1500.0,
        )
        monkeypatch.setattr(
            inference_features,
            "_get_latest_computed_features",
            lambda s, fid: {},
        )
        monkeypatch.setattr(
            inference_features,
            "_get_cached_odds",
            lambda s, a, b, d: (None, None),
        )
        monkeypatch.setattr(
            inference_features,
            "_query_ref_state",
            _stub_query_ref_state,
        )
        # Plan 23-03 META stubs — safe defaults so MagicMock session
        # doesn't break META queries (this parity test only checks REF).
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

        def _stub_fighter(fid: int):
            return MagicMock(
                id=fid,
                name=f"Fighter-{fid}",
                height_inches=70.0,
                reach_inches=72.0,
                leg_reach_inches=40.0,
                stance="Orthodox",
                date_of_birth=_date(1990, 1, 1),
            )

        session = MagicMock()

        # ── THE PARITY GATE ────────────────────────────────────────────────
        # For each fight in the fixture, build a row via inference and
        # compare its REF cols (indices 72-74) to the training-path REF cols
        # for the same fight. They MUST be byte-identical to within float
        # roundoff (Pitfall #12 LIVE-03 parity).
        for fight in fight_records:
            ref_id = fight["referee_id"]
            event_date = fight["event_date"]
            fa = _stub_fighter(fight["fighter_a_id"])
            fb = _stub_fighter(fight["fighter_b_id"])

            inf_row = inference_features.build(
                session,
                fa,
                fb,
                event_date,
                feature_set="v2.2",
                referee_id=ref_id,
            )[0]

            # Find the corresponding training-path row by fight_id.
            train_row_idx = next(
                i for i, f in enumerate(fight_records) if f["fight_id"] == fight["fight_id"]
            )
            train_row = X_train[train_row_idx]

            for col_idx, col_name in (
                (72, "ref_finish_rate_shrunk"),
                (73, "ref_decision_rate_shrunk"),
                (74, "ref_no_action_rate_shrunk"),
            ):
                train_val = train_row[col_idx]
                inf_val = inf_row[col_idx]
                assert train_val == _pytest.approx(inf_val, abs=1e-12), (
                    f"LIVE-03 parity violation at fight {fight['fight_id']}, "
                    f"col {col_name} (idx {col_idx}): "
                    f"train={train_val} inference={inf_val}"
                )

        # Sanity guard: REF cols should NOT all be identical across fights
        # (otherwise the parity check is trivially passing on Bayesian
        # fallback). Fight 203's ref=1 has a prior finish from fight 201, so
        # its REF cols should differ from fight 201's (no history at the
        # point of fight 201).
        assert not np.allclose(X_train[0, 72:75], X_train[2, 72:75]), (
            "Training-path REF cols are suspiciously identical between "
            "fight 201 (no history) and fight 203 (1 prior finish for "
            "ref=1) — the parity test may be trivially passing."
        )


# ── Plan 23-02 TRAVEL helper tests ──────────────────────────────────────────


class TestHaversine:
    """Plan 23-02 Tests 1-3 + 4: pure-Python haversine_miles."""

    def test_haversine_zero_distance(self) -> None:
        """Test 1: haversine_miles at same point returns 0.0."""
        from ufc_prediction.ml.features_v22 import haversine_miles

        assert haversine_miles(0, 0, 0, 0) == pytest.approx(0.0)
        assert haversine_miles(40.0, -74.0, 40.0, -74.0) == pytest.approx(0.0)

    def test_haversine_nyc_to_la(self) -> None:
        """Test 2: NYC→LA ≈ 2451 miles (tolerance ±15 mi)."""
        from ufc_prediction.ml.features_v22 import haversine_miles

        # NYC (40.7128, -74.0060) → LA (34.0522, -118.2437)
        dist = haversine_miles(40.7128, -74.0060, 34.0522, -118.2437)
        assert abs(dist - 2451) < 15, f"Expected ~2451 mi NYC→LA, got {dist}"

    def test_haversine_symmetric(self) -> None:
        """Test 3: haversine_miles(a,b,c,d) == haversine_miles(c,d,a,b)."""
        from ufc_prediction.ml.features_v22 import haversine_miles

        a = haversine_miles(40.7128, -74.0060, 34.0522, -118.2437)
        b = haversine_miles(34.0522, -118.2437, 40.7128, -74.0060)
        assert a == pytest.approx(b)


def test_no_banned_runtime_imports() -> None:
    """Plan 23-02 Test 4 + 14 (D-05): no geopy/pytz/timezonefinder runtime imports.

    Greps the features_v22 submodule for banned runtime imports per CONTEXT
    D-05. xtra patterns banned: geopy, pytz, timezonefinder (PyPI deps that
    Phase 22 used one-off for backfill but are NOT runtime deps).
    """
    import subprocess

    result = subprocess.run(
        [
            "grep",
            "-rn",
            "from geopy\\|import geopy\\|from pytz\\|import pytz\\|from timezonefinder\\|import timezonefinder",
            "src/ufc_prediction/ml/features_v22/",
        ],
        capture_output=True,
        text=True,
    )
    # grep exit code 1 = no matches (the desired state)
    assert result.returncode == 1, (
        f"D-05 ban violated — banned runtime imports found:\n{result.stdout}"
    )


class TestTzOffsetHours:
    """Plan 23-02 Tests 5-8: zoneinfo UTC-offset lookup at event_date.

    Pitfall #4: DST evaluated at event_date (not today). Naive datetime at
    midnight of event_date.
    """

    def test_tz_offset_pdt(self) -> None:
        """Test 5: LA in June 2026 → -7.0 (PDT, daylight saving)."""
        from ufc_prediction.ml.features_v22 import tz_offset_hours

        assert tz_offset_hours("America/Los_Angeles", date(2026, 6, 14)) == -7.0

    def test_tz_offset_pst(self) -> None:
        """Test 6: LA in January 2026 → -8.0 (PST, standard time)."""
        from ufc_prediction.ml.features_v22 import tz_offset_hours

        assert tz_offset_hours("America/Los_Angeles", date(2026, 1, 14)) == -8.0

    def test_tz_offset_dst_boundary(self) -> None:
        """Test 7: NY DST boundary Nov 1/2 2026.

        US DST ends 2026-11-01 at 02:00 local. With midnight-based naive
        datetime evaluation:
            2026-11-01 00:00 ET → still EDT (-4)
            2026-11-02 00:00 ET → EST (-5) after the transition
        """
        from ufc_prediction.ml.features_v22 import tz_offset_hours

        # Day before DST ends: midnight is still EDT
        assert tz_offset_hours("America/New_York", date(2026, 11, 1)) == -4.0
        # Day after DST ends: now EST
        assert tz_offset_hours("America/New_York", date(2026, 11, 2)) == -5.0

    def test_tz_offset_tokyo(self) -> None:
        """Test 8: Tokyo summer → +9.0 (Japan has no DST)."""
        from ufc_prediction.ml.features_v22 import tz_offset_hours

        assert tz_offset_hours("Asia/Tokyo", date(2026, 6, 14)) == 9.0
        # And winter — same offset (no DST).
        assert tz_offset_hours("Asia/Tokyo", date(2026, 1, 14)) == 9.0


class TestComputeTzShiftSigned:
    """Plan 23-02 Tests 9-11: signed continuous tz_shift in hours."""

    def test_compute_tz_shift_first_fight(self) -> None:
        """Test 9: None prior → 0.0 (sentinel per D-04)."""
        from ufc_prediction.ml.features_v22 import compute_tz_shift_signed

        out = compute_tz_shift_signed(
            None,
            None,
            "America/Las_Vegas",
            date(2026, 5, 1),
        )
        assert out == 0.0

        # Either being None triggers sentinel.
        out2 = compute_tz_shift_signed(
            "America/Las_Vegas",
            None,
            "Asia/Tokyo",
            date(2026, 5, 1),
        )
        assert out2 == 0.0

        out3 = compute_tz_shift_signed(
            None,
            date(2026, 1, 1),
            "Asia/Tokyo",
            date(2026, 5, 1),
        )
        assert out3 == 0.0

    def test_compute_tz_shift_eastward_positive(self) -> None:
        """Test 10: LA→Tokyo = positive (traveling east; PDT -7h → JST +9h)."""
        from ufc_prediction.ml.features_v22 import compute_tz_shift_signed

        # 2026-05-01 LA: PDT = -7. 2026-05-08 Tokyo: JST = +9.
        # shift = curr_offset - prior_offset = 9 - (-7) = +16
        out = compute_tz_shift_signed(
            "America/Los_Angeles",
            date(2026, 5, 1),
            "Asia/Tokyo",
            date(2026, 5, 8),
        )
        assert out == pytest.approx(16.0)

    def test_compute_tz_shift_westward_negative(self) -> None:
        """Test 11: Tokyo→LA = negative (traveling west; JST +9h → PDT -7h)."""
        from ufc_prediction.ml.features_v22 import compute_tz_shift_signed

        # shift = -7 - 9 = -16
        out = compute_tz_shift_signed(
            "Asia/Tokyo",
            date(2026, 5, 1),
            "America/Los_Angeles",
            date(2026, 5, 8),
        )
        assert out == pytest.approx(-16.0)


class TestComputeTravelFeatures:
    """Plan 23-02 Tests 12-13: composed haversine + tz_shift for one fighter."""

    def test_compute_travel_features_first_fight(self) -> None:
        """Test 12: prior_venue=None → both values = 0 sentinel (D-04)."""
        from ufc_prediction.ml.features_v22 import compute_travel_features

        out = compute_travel_features(
            None,
            {"lat": 36.1, "lon": -115.2, "timezone_iana": "America/Los_Angeles"},
            date(2026, 5, 1),
        )
        assert out == {"travel_distance_miles": 0.0, "tz_shift_signed": 0.0}

    def test_compute_travel_features_subsequent(self) -> None:
        """Test 13: NYC→LA: ≈2451 mi haversine + tz_shift = PST(-8) - EST(-5) = -3."""
        from ufc_prediction.ml.features_v22 import compute_travel_features

        # Prior fight: NYC, 2026-01-01 → EST = -5h.
        # Current fight: LA, 2026-03-01 → PST = -8h (DST starts 2026-03-08).
        out = compute_travel_features(
            {
                "lat": 40.7128,
                "lon": -74.0060,
                "timezone_iana": "America/New_York",
                "event_date": date(2026, 1, 1),
            },
            {
                "lat": 34.0522,
                "lon": -118.2437,
                "timezone_iana": "America/Los_Angeles",
            },
            date(2026, 3, 1),
        )
        # Haversine NYC→LA ≈ 2451 mi
        assert abs(out["travel_distance_miles"] - 2451) < 15
        # tz_shift = -8 - (-5) = -3 (traveling west)
        assert out["tz_shift_signed"] == pytest.approx(-3.0)


# ── TestTrainInferenceParity::test_travel_cols_parity (LIVE-03 D-10) ────────


class TestTrainInferenceParityTravel:
    """Phase 23 Plan 23-02 D-10 — TRAVEL cols byte-identical across paths.

    The LIVE-03 parity gate for TRAVEL features. Both feature_matrix.assemble(
    ..., feature_set='v2.2') and inference_features.build(..., feature_set='v2.2')
    must emit byte-identical values for the 6 TRAVEL cols (indices 75-80)
    when fed the same fixture data (same fighters, same venues, same dates).

    Architecture: both paths call into
    `features_v22.travel.compute_travel_features` (single source of truth per
    Pitfall #12); they differ only in how they BUILD the venue/prior-venue
    inputs. This test feeds the same in-memory venue lookups into both paths
    and asserts byte-identical results.
    """

    def test_travel_cols_parity_between_train_and_inference(
        self,
        monkeypatch,
    ) -> None:
        """Same fixture → byte-identical TRAVEL cols across train + inference."""
        from datetime import date as _date
        from unittest.mock import MagicMock

        import numpy as np
        import pytest as _pytest

        from ufc_prediction.ml import inference_features
        from ufc_prediction.ml.feature_matrix import (
            FeatureMatrixAssembler,
            _build_event_venue_lookup,
            _build_fighter_prior_venues,
        )

        # ── Shared fixture: 3 fights, 3 venues (NYC, LA, Tokyo) ────────────
        venue_nyc = {
            "lat": 40.7128,
            "lon": -74.0060,
            "timezone_iana": "America/New_York",
        }
        venue_la = {
            "lat": 34.0522,
            "lon": -118.2437,
            "timezone_iana": "America/Los_Angeles",
        }
        venue_tokyo = {
            "lat": 35.6762,
            "lon": 139.6503,
            "timezone_iana": "Asia/Tokyo",
        }
        event_venue_map = {201: venue_nyc, 202: venue_la, 203: venue_tokyo}

        fight_records = [
            {
                "fight_id": 201,
                "event_id": 201,
                "event_date": _date(2024, 1, 1),
                "fighter_a_id": 1,
                "fighter_b_id": 2,
                "winner_id": 1,
                "weight_class": "Lightweight",
                "referee_id": None,
                "method": "KO/TKO",
                "venue_lat": venue_nyc["lat"],
                "venue_lon": venue_nyc["lon"],
                "venue_timezone_iana": venue_nyc["timezone_iana"],
            },
            {
                "fight_id": 202,
                "event_id": 202,
                "event_date": _date(2024, 6, 1),
                "fighter_a_id": 1,
                "fighter_b_id": 3,
                "winner_id": 1,
                "weight_class": "Lightweight",
                "referee_id": None,
                "method": "Decision - Unanimous",
                "venue_lat": venue_la["lat"],
                "venue_lon": venue_la["lon"],
                "venue_timezone_iana": venue_la["timezone_iana"],
            },
            {
                "fight_id": 203,
                "event_id": 203,
                "event_date": _date(2025, 3, 1),
                "fighter_a_id": 1,
                "fighter_b_id": 4,
                "winner_id": 1,
                "weight_class": "Lightweight",
                "referee_id": None,
                "method": "Submission",
                "venue_lat": venue_tokyo["lat"],
                "venue_lon": venue_tokyo["lon"],
                "venue_timezone_iana": venue_tokyo["timezone_iana"],
            },
        ]

        # ── Minimal Elo / computed / physical / division fixtures ──────────
        elo_default = {
            "elo_overall": 1500.0,
            "elo_striking": 1500.0,
            "elo_grappling": 1500.0,
        }
        elo_features = {
            (fid, fight["fight_id"]): dict(elo_default)
            for fight in fight_records
            for fid in (fight["fighter_a_id"], fight["fighter_b_id"])
        }
        computed_features: dict = {}
        fighter_physicals = {
            fid: {
                "height_inches": 70.0,
                "reach_inches": 72.0,
                "leg_reach_inches": 40.0,
                "stance": "Orthodox",
                "date_of_birth": _date(1990, 1, 1),
            }
            for fid in (1, 2, 3, 4)
        }
        division_medians = {
            "Lightweight": {
                "height_inches": 70.0,
                "reach_inches": 72.0,
                "leg_reach_inches": 40.0,
            },
        }

        # ── Build shared venue lookups (training path's pre-pass output) ───
        sorted_records = sorted(fight_records, key=lambda f: f["event_date"])
        shared_event_venues = _build_event_venue_lookup(sorted_records)
        shared_prior_venues = _build_fighter_prior_venues(
            sorted_records,
            shared_event_venues,
        )

        # ── Training path: pass shared inputs explicitly ───────────────────
        assembler = FeatureMatrixAssembler()
        X_train, _, _ = assembler.assemble(
            fight_records,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
            feature_set="v2.2",
            event_venues=shared_event_venues,
            fighter_prior_venues=shared_prior_venues,
        )

        # ── Inference path stubs: return the SAME venue lookups ────────────
        def _stub_current_venue(session, event_id):
            return event_venue_map.get(event_id)

        def _stub_prior_venue(session, fighter_id, before_date):
            # Walk shared_prior_venues and find the latest match for this
            # fighter strictly before before_date. The training-path
            # _build_fighter_prior_venues uses snapshot-before-update so a
            # given (fighter_id, fight_id) key gives the prior venue for
            # THAT fight; we need the one keyed by (fighter_id, the fight
            # at before_date). Easier: walk fight_records ourselves.
            candidates = []
            for f in fight_records:
                if f["event_date"] >= before_date:
                    continue
                if fighter_id in (f["fighter_a_id"], f["fighter_b_id"]):
                    venue = event_venue_map[f["event_id"]]
                    candidates.append((f["event_date"], venue))
            if not candidates:
                return None
            candidates.sort(key=lambda x: x[0], reverse=True)
            _ed, v = candidates[0]
            return {**v, "event_date": _ed}

        monkeypatch.setattr(
            inference_features,
            "_get_latest_elo",
            lambda s, fid, et: 1500.0,
        )
        monkeypatch.setattr(
            inference_features,
            "_get_latest_computed_features",
            lambda s, fid: {},
        )
        monkeypatch.setattr(
            inference_features,
            "_get_cached_odds",
            lambda s, a, b, d: (None, None),
        )
        monkeypatch.setattr(
            inference_features,
            "_query_ref_state",
            lambda s, r, d: ({}, {"finish": 0.0, "decision": 0.0, "no_action": 0.0}),
        )
        monkeypatch.setattr(
            inference_features,
            "_query_current_venue",
            _stub_current_venue,
        )
        monkeypatch.setattr(
            inference_features,
            "_query_fighter_prior_venue",
            _stub_prior_venue,
        )
        # Plan 23-03 META stubs — this parity test only checks TRAVEL,
        # safe defaults prevent MagicMock session from breaking META queries.
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

        def _stub_fighter(fid: int):
            return MagicMock(
                id=fid,
                name=f"Fighter-{fid}",
                height_inches=70.0,
                reach_inches=72.0,
                leg_reach_inches=40.0,
                stance="Orthodox",
                date_of_birth=_date(1990, 1, 1),
            )

        session = MagicMock()

        # ── THE PARITY GATE: TRAVEL cols 75-80 byte-identical ──────────────
        # Note: cross-path identity must account for the deterministic A/B
        # swap in feature_matrix.assemble() — the same swap is NOT applied
        # in inference_features.build() (fa is always red, fb is always
        # blue at inference). Therefore we feed the INFERENCE path the
        # fighters in the SAME order the training path uses POST-swap.
        import hashlib

        for fight in fight_records:
            fight_id = fight["fight_id"]
            event_date_val = fight["event_date"]
            # Replicate the deterministic swap from feature_matrix.assemble.
            swap = int(hashlib.md5(str(fight_id).encode()).hexdigest(), 16) % 2 == 0
            if swap:
                a_id = fight["fighter_b_id"]
                b_id = fight["fighter_a_id"]
            else:
                a_id = fight["fighter_a_id"]
                b_id = fight["fighter_b_id"]

            fa = _stub_fighter(a_id)
            fb = _stub_fighter(b_id)

            inf_row = inference_features.build(
                session,
                fa,
                fb,
                event_date_val,
                feature_set="v2.2",
                event_id=fight["event_id"],
            )[0]

            train_row_idx = next(
                i for i, f in enumerate(fight_records) if f["fight_id"] == fight_id
            )
            train_row = X_train[train_row_idx]

            for col_idx, col_name in (
                (75, "travel_distance_miles_red"),
                (76, "travel_distance_miles_blue"),
                (77, "travel_distance_miles_diff"),
                (78, "tz_shift_red_signed"),
                (79, "tz_shift_blue_signed"),
                (80, "tz_shift_diff_signed"),
            ):
                train_val = train_row[col_idx]
                inf_val = inf_row[col_idx]
                assert train_val == _pytest.approx(inf_val, abs=1e-12), (
                    f"LIVE-03 TRAVEL parity violation at fight {fight_id}, "
                    f"col {col_name} (idx {col_idx}): "
                    f"train={train_val} inference={inf_val}"
                )

        # Sanity guard: TRAVEL cols should NOT all be zero (otherwise the
        # parity check is trivially passing on the debut sentinel).
        # Fight 202 has fighter 1 traveling NYC→LA → red_dist ≈ 2451.
        idx_202 = 1
        assert not np.allclose(X_train[idx_202, 75:78], 0.0), (
            "Training-path TRAVEL distance cols suspiciously all-zero at "
            "fight 202 (fighter 1 should have travelled NYC→LA)."
        )


# ── Plan 23-03 META helper tests ────────────────────────────────────────────


class TestLayoffDays:
    """Plan 23-03 Tests 1-4: per-fighter layoff with D-06 720-day clip."""

    def test_layoff_days_first_fight(self) -> None:
        """Test 1: prior=None → 0.0 (first-fight sentinel, NOT NaN)."""
        from ufc_prediction.ml.features_v22 import layoff_days

        assert layoff_days(date(2026, 5, 1), None) == 0.0

    def test_layoff_days_normal(self) -> None:
        """Test 2: 89 days between fights."""
        from ufc_prediction.ml.features_v22 import layoff_days

        assert layoff_days(date(2026, 5, 1), date(2026, 2, 1)) == 89.0

    def test_layoff_days_clipped_at_720(self) -> None:
        """Test 3: > 720 days clipped at 720 (D-06)."""
        from ufc_prediction.ml.features_v22 import layoff_days

        # 2023-01-01 to 2026-05-01 = ~1216 days → clipped to 720
        assert layoff_days(date(2026, 5, 1), date(2023, 1, 1)) == 720.0

    def test_layoff_days_custom_clip(self) -> None:
        """Test 4: clip_max kwarg overrides default."""
        from ufc_prediction.ml.features_v22 import layoff_days

        assert (
            layoff_days(
                date(2026, 5, 1),
                date(2023, 1, 1),
                clip_max=365,
            )
            == 365.0
        )


class TestAgeAtFight:
    """Plan 23-03 Tests 5-7: age in years, uses fight_date (Pitfall #5)."""

    def test_age_at_fight_known(self) -> None:
        """Test 5: 1990-01-01 born, 2026-01-01 fight → 36.0 ± 0.01 years."""
        from ufc_prediction.ml.features_v22 import age_at_fight

        age = age_at_fight(date(1990, 1, 1), date(2026, 1, 1))
        # (2026-01-01 - 1990-01-01).days / 365.25 = 13149 / 365.25 ≈ 35.997...
        expected = (date(2026, 1, 1) - date(1990, 1, 1)).days / 365.25
        assert age == pytest.approx(expected, abs=1e-9)
        assert abs(age - 36.0) < 0.01

    def test_age_at_fight_none_dob(self) -> None:
        """Test 6: None birth_date → NaN (Pattern D)."""
        import math as _math

        from ufc_prediction.ml.features_v22 import age_at_fight

        result = age_at_fight(None, date(2026, 1, 1))
        assert _math.isnan(result)

    def test_age_at_fight_uses_fight_date_not_today(self) -> None:
        """Test 7: age is deterministic for any fight_date (Pitfall #5 guard).

        This is the regression test that prevents date.today() from sneaking
        back into the age computation. A fighter born 1990-01-01 in a fight
        dated 2020-01-01 is ~30 years old — independent of whatever date
        the test happens to run on.
        """
        from ufc_prediction.ml.features_v22 import age_at_fight

        age = age_at_fight(date(1990, 1, 1), date(2020, 1, 1))
        expected = (date(2020, 1, 1) - date(1990, 1, 1)).days / 365.25
        assert age == pytest.approx(expected, abs=1e-9)
        # Independent of today's date.
        assert abs(age - 30.0) < 0.01


class TestEloVelocity:
    """Plan 23-03 Tests 8-11: rate-of-change over N prior fights."""

    def test_elo_velocity_normal(self) -> None:
        """Test 8: 6-entry history, window=5 → (1550 - 1500) / 5 = 10.0.

        Slice math (per plan's pseudo + meta.py impl):
            ``(elo_history[-1] - elo_history[-(window + 1)]) / window``
        With 6 entries and window=5: ``-(5+1) = -6`` → index 0 (oldest).
        So ``(1550 - 1500) / 5 = 10.0``.
        """
        from ufc_prediction.ml.features_v22 import elo_velocity

        result = elo_velocity(
            [1500, 1510, 1520, 1530, 1540, 1550],
            window=5,
        )
        assert result == pytest.approx(10.0)

    def test_elo_velocity_insufficient_history(self) -> None:
        """Test 9: < window+1 entries → NaN (Pitfall #7)."""
        import math as _math

        from ufc_prediction.ml.features_v22 import elo_velocity

        # 3 entries, window=5 → need 6 → NaN
        assert _math.isnan(elo_velocity([1500, 1510, 1520], window=5))

    def test_elo_velocity_empty_history(self) -> None:
        """Test 10: empty history → NaN."""
        import math as _math

        from ufc_prediction.ml.features_v22 import elo_velocity

        assert _math.isnan(elo_velocity([], window=5))

    def test_elo_velocity_window_3(self) -> None:
        """Test 11: 4-entry history, window=3 → (1530 - 1500) / 3 = 10.0.

        Slice math: elo_history[-1] - elo_history[-(window+1)] / window
        With window=3 and 4 entries: result is (1530 - 1500) / 3 = 10.0.
        """
        from ufc_prediction.ml.features_v22 import elo_velocity

        result = elo_velocity([1500, 1510, 1520, 1530], window=3)
        assert result == pytest.approx(10.0)


class TestDivisionFinishRateShrunk:
    """Plan 23-03 Tests 12-16: Beta-binomial EB per-division finish rate.

    REUSES `beta_binomial_shrink` + `classify_outcome` from features_v22.ref
    (DRY per CONTEXT D-06).
    """

    def test_division_finish_rate_unknown_class(self) -> None:
        """Test 12: weight_class=None → Bayesian fallback to global_rate."""
        from ufc_prediction.ml.features_v22 import division_finish_rate_shrunk

        result = division_finish_rate_shrunk(
            None,
            date(2026, 1, 1),
            {},
            global_finish_rate=0.55,
        )
        assert result == pytest.approx(0.55)

    def test_division_finish_rate_no_history(self) -> None:
        """Test 13: class known but no prior fights → collapse to global."""
        from ufc_prediction.ml.features_v22 import division_finish_rate_shrunk

        result = division_finish_rate_shrunk(
            "Lightweight",
            date(2026, 1, 1),
            {"Lightweight": []},
            global_finish_rate=0.55,
        )
        assert result == pytest.approx(0.55)

    def test_division_finish_rate_with_history(self) -> None:
        """Test 14: 10 Lightweight priors (7 finishes) → EB shrunk rate.

        (7 + 50*0.55) / (10 + 50) = (7 + 27.5) / 60 = 34.5 / 60 ≈ 0.575
        """
        from ufc_prediction.ml.features_v22 import division_finish_rate_shrunk

        history = {
            "Lightweight": [
                {"event_date": date(2025, 1, 1), "method": "KO/TKO"},
                {"event_date": date(2025, 2, 1), "method": "Submission"},
                {"event_date": date(2025, 3, 1), "method": "KO/TKO"},
                {"event_date": date(2025, 4, 1), "method": "TKO"},
                {"event_date": date(2025, 5, 1), "method": "Submission"},
                {"event_date": date(2025, 6, 1), "method": "KO/TKO"},
                {"event_date": date(2025, 7, 1), "method": "SUB"},
                {"event_date": date(2025, 8, 1), "method": "Decision - Unanimous"},
                {"event_date": date(2025, 9, 1), "method": "Decision - Split"},
                {"event_date": date(2025, 10, 1), "method": "Decision - Majority"},
            ],
        }
        result = division_finish_rate_shrunk(
            "Lightweight",
            date(2026, 1, 1),
            history,
            global_finish_rate=0.55,
            k_shrink=50.0,
        )
        expected = (7 + 50 * 0.55) / (10 + 50)
        assert result == pytest.approx(expected, abs=1e-9)

    def test_division_finish_rate_temporal_cutoff(self) -> None:
        """Test 15: post-as-of-date fights do NOT contribute (Pitfall #4)."""
        from ufc_prediction.ml.features_v22 import division_finish_rate_shrunk

        # All KO/TKO finishes — but 2/3 are AFTER the cutoff.
        history = {
            "Lightweight": [
                {"event_date": date(2025, 1, 1), "method": "KO/TKO"},
                # These two are AFTER as_of_date=2025-06-01 — must NOT count.
                {"event_date": date(2025, 7, 1), "method": "KO/TKO"},
                {"event_date": date(2026, 1, 1), "method": "Submission"},
            ],
        }
        result = division_finish_rate_shrunk(
            "Lightweight",
            date(2025, 6, 1),
            history,
            global_finish_rate=0.5,
            k_shrink=50.0,
        )
        # Only 1 fight contributes (1 finish, n=1):
        # (1 + 50*0.5) / (1 + 50) = 26 / 51 ≈ 0.5098...
        expected = (1 + 50 * 0.5) / (1 + 50)
        assert result == pytest.approx(expected, abs=1e-9)

    def test_division_finish_rate_reuses_ref_helper(self) -> None:
        """Test 16: meta.py imports beta_binomial_shrink from ref.py (DRY/D-06)."""
        import subprocess

        result = subprocess.run(
            [
                "grep",
                "-c",
                "from ufc_prediction.ml.features_v22.ref import",
                "src/ufc_prediction/ml/features_v22/meta.py",
            ],
            capture_output=True,
            text=True,
        )
        # grep -c prints the count; require >= 1 import line.
        count = int(result.stdout.strip()) if result.returncode == 0 else 0
        assert count >= 1, (
            f"meta.py must import beta_binomial_shrink from ref.py (DRY per "
            f"D-06); got {count} matching import lines"
        )


class TestReachDiffNormalized:
    """Plan 23-03 Tests 17-20: reach diff scaled by mean reach in division."""

    def test_reach_diff_normalized_normal(self) -> None:
        """Test 17: (74 - 72) / 73 ≈ 0.0274."""
        from ufc_prediction.ml.features_v22 import reach_diff_normalized

        result = reach_diff_normalized(74.0, 72.0, 73.0)
        assert result == pytest.approx(2.0 / 73.0, abs=1e-9)

    def test_reach_diff_normalized_none_a(self) -> None:
        """Test 18: None reach_a → NaN."""
        import math as _math

        from ufc_prediction.ml.features_v22 import reach_diff_normalized

        assert _math.isnan(reach_diff_normalized(None, 72.0, 73.0))

    def test_reach_diff_normalized_none_b(self) -> None:
        """Test 19: None reach_b → NaN."""
        import math as _math

        from ufc_prediction.ml.features_v22 import reach_diff_normalized

        assert _math.isnan(reach_diff_normalized(74.0, None, 73.0))

    def test_reach_diff_normalized_zero_mean(self) -> None:
        """Test 20: division_mean=0 → NaN (avoid div-by-zero)."""
        import math as _math

        from ufc_prediction.ml.features_v22 import reach_diff_normalized

        assert _math.isnan(reach_diff_normalized(74.0, 72.0, 0.0))


class TestFeatureColumnsV22Dedup:
    """Plan 23-03 Tests 21-22: META section dedup vs FEATURE_COLUMNS_NO_NET.

    Per CONTEXT D-07 + RESEARCH Q4 RESOLVED:
        - No META col may overlap FEATURE_COLUMNS_NO_NET.
        - `layoff_days_diff` is DROPPED (semantic dup of
          FEATURE_COLUMNS_NO_NET[61] `days_since_last_fight_diff`).
    """

    def test_meta_cols_no_duplicates_with_no_net(self) -> None:
        """Test 21: META slice [81:90] disjoint from NO_NET (D-07 dedup)."""
        from ufc_prediction.ml.config import (
            FEATURE_COLUMNS_NO_NET,
            FEATURE_COLUMNS_V22,
        )

        meta_slice = FEATURE_COLUMNS_V22[81:90]
        overlap = set(meta_slice) & set(FEATURE_COLUMNS_NO_NET)
        assert overlap == set(), f"D-07 dedup violation in META section [81:90]: {overlap}"

    def test_meta_section_excludes_layoff_days_diff(self) -> None:
        """Test 22: layoff_days_diff absent (Q4 RESOLVED + B-1 regression)."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS_V22

        # Not in META slice
        assert "layoff_days_diff" not in FEATURE_COLUMNS_V22[81:90]
        # Not anywhere in v2.2
        assert "layoff_days_diff" not in FEATURE_COLUMNS_V22


# ── TestTrainInferenceParity::test_meta_cols_parity (LIVE-03 D-10) ──────────


class TestTrainInferenceParityMeta:
    """Plan 23-03 D-10 — META cols byte-identical across train + inference.

    LIVE-03 parity gate for META features. Both
    `feature_matrix.assemble(..., feature_set='v2.2')` and
    `inference_features.build(..., feature_set='v2.2')` must emit byte-
    identical values for the 9 META cols (indices 81-89) when fed the same
    fixture data.

    Architecture: both paths call into `features_v22.meta` helpers (single
    source of truth per Pitfall #12); they differ only in how they BUILD the
    DoB / reach / Elo-history / division-history inputs.
    """

    def test_meta_cols_parity_between_train_and_inference(
        self,
        monkeypatch,
    ) -> None:
        """Same fixture → byte-identical META cols across train + inference."""
        from datetime import date as _date
        from unittest.mock import MagicMock

        import pytest as _pytest

        from ufc_prediction.ml import inference_features
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        # ── Shared fixture: 3 fights, 4 fighters with full DoB/reach ────────
        # Fighter 1: born 1990-01-01, reach 74
        # Fighter 2: born 1992-06-01, reach 72
        # Fighter 3: born 1988-03-15, reach 76
        # Fighter 4: born 1995-09-09, reach 70
        fighter_dob = {
            1: _date(1990, 1, 1),
            2: _date(1992, 6, 1),
            3: _date(1988, 3, 15),
            4: _date(1995, 9, 9),
        }
        fighter_reach = {1: 74.0, 2: 72.0, 3: 76.0, 4: 70.0}

        fight_records = [
            {
                "fight_id": 301,
                "event_id": 301,
                "event_date": _date(2024, 1, 1),
                "fighter_a_id": 1,
                "fighter_b_id": 2,
                "winner_id": 1,
                "weight_class": "Lightweight",
                "referee_id": None,
                "method": "KO/TKO",
            },
            {
                "fight_id": 302,
                "event_id": 302,
                "event_date": _date(2024, 6, 1),
                "fighter_a_id": 3,
                "fighter_b_id": 4,
                "winner_id": 3,
                "weight_class": "Lightweight",
                "referee_id": None,
                "method": "Decision - Unanimous",
            },
            {
                "fight_id": 303,
                "event_id": 303,
                "event_date": _date(2025, 3, 1),
                "fighter_a_id": 1,
                "fighter_b_id": 3,
                "winner_id": 1,
                "weight_class": "Lightweight",
                "referee_id": None,
                "method": "Submission",
            },
        ]

        # Minimal Elo / computed / physical fixtures (per pattern from REF/
        # TRAVEL tests).
        elo_default = {
            "elo_overall": 1500.0,
            "elo_striking": 1500.0,
            "elo_grappling": 1500.0,
        }
        elo_features = {
            (fid, fight["fight_id"]): dict(elo_default)
            for fight in fight_records
            for fid in (fight["fighter_a_id"], fight["fighter_b_id"])
        }
        computed_features: dict = {}
        fighter_physicals = {
            fid: {
                "height_inches": 70.0,
                "reach_inches": fighter_reach[fid],
                "leg_reach_inches": 40.0,
                "stance": "Orthodox",
                "date_of_birth": fighter_dob[fid],
            }
            for fid in (1, 2, 3, 4)
        }
        division_medians = {
            "Lightweight": {
                "height_inches": 70.0,
                "reach_inches": 73.0,
                "leg_reach_inches": 40.0,
            },
        }

        # ── Training path: assemble v2.2 X matrix ───────────────────────────
        assembler = FeatureMatrixAssembler()
        X_train, _, _ = assembler.assemble(
            fight_records,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
            feature_set="v2.2",
        )

        # ── Inference-path stubs: provide same META state via DB queries ───
        def _stub_fighter_prior_fight_date(session, fighter_id, before_date):
            """Most-recent prior fight (strict < before_date) for fighter."""
            candidates = []
            for f in fight_records:
                if f["event_date"] >= before_date:
                    continue
                if fighter_id in (f["fighter_a_id"], f["fighter_b_id"]):
                    candidates.append(f["event_date"])
            if not candidates:
                return None
            return max(candidates)

        def _stub_elo_history(session, fighter_id, before_date, limit):
            """Return list of {"elo_overall", "elo_striking", "elo_grappling"}
            for fighter, strict < before_date, chronological."""
            history = []
            for f in sorted(
                fight_records,
                key=lambda x: x["event_date"],
            ):
                if f["event_date"] >= before_date:
                    continue
                if fighter_id in (f["fighter_a_id"], f["fighter_b_id"]):
                    elo = elo_features.get(
                        (fighter_id, f["fight_id"]),
                        elo_default,
                    )
                    history.append(
                        {
                            "elo_overall": elo["elo_overall"],
                            "elo_striking": elo["elo_striking"],
                            "elo_grappling": elo["elo_grappling"],
                        }
                    )
            return history[-limit:]

        def _stub_division_state(session, weight_class, before_date):
            """Return ({wc: [{event_date, method}]}, global_finish_rate).

            Per training-path discipline: per-class history is the FULL
            chronological list (helper applies temporal cutoff inside);
            global rate is corpus-wide (no temporal filter — matches
            feature_matrix._build_division_history).
            """
            from ufc_prediction.ml.features_v22.ref import classify_outcome

            div_hist: dict = {}
            total_finishes = 0
            total = 0
            for f in fight_records:
                wc = f.get("weight_class")
                if wc is not None:
                    div_hist.setdefault(wc, []).append(
                        {
                            "event_date": f["event_date"],
                            "method": f.get("method"),
                        }
                    )
                total += 1
                if classify_outcome(f.get("method")) == "finish":
                    total_finishes += 1
            g = total_finishes / total if total > 0 else 0.0
            return div_hist, g

        def _stub_division_mean_reach(session, weight_class):
            return division_medians.get(weight_class, {}).get("reach_inches")

        def _stub_fighter_division(session, fighter_id):
            return "Lightweight"

        # Inject all META DB stubs.
        monkeypatch.setattr(
            inference_features,
            "_query_fighter_prior_fight_date",
            _stub_fighter_prior_fight_date,
        )
        monkeypatch.setattr(
            inference_features,
            "_query_elo_history",
            _stub_elo_history,
        )
        monkeypatch.setattr(
            inference_features,
            "_query_division_state",
            _stub_division_state,
        )
        monkeypatch.setattr(
            inference_features,
            "_query_division_mean_reach",
            _stub_division_mean_reach,
        )
        monkeypatch.setattr(
            inference_features,
            "_query_fighter_division",
            _stub_fighter_division,
        )

        # Stub other DB calls so we don't hit Postgres.
        monkeypatch.setattr(
            inference_features,
            "_get_latest_elo",
            lambda s, fid, et: 1500.0,
        )
        monkeypatch.setattr(
            inference_features,
            "_get_latest_computed_features",
            lambda s, fid: {},
        )
        monkeypatch.setattr(
            inference_features,
            "_get_cached_odds",
            lambda s, a, b, d: (None, None),
        )
        monkeypatch.setattr(
            inference_features,
            "_query_ref_state",
            lambda s, r, d: ({}, {"finish": 0.0, "decision": 0.0, "no_action": 0.0}),
        )
        # No venue lookups for this fixture (TRAVEL cols can be NaN here).
        monkeypatch.setattr(
            inference_features,
            "_query_current_venue",
            lambda s, eid: None,
        )
        monkeypatch.setattr(
            inference_features,
            "_query_fighter_prior_venue",
            lambda s, fid, bd: None,
        )

        def _stub_fighter_orm(fid: int):
            return MagicMock(
                id=fid,
                name=f"Fighter-{fid}",
                height_inches=70.0,
                reach_inches=fighter_reach[fid],
                leg_reach_inches=40.0,
                stance="Orthodox",
                date_of_birth=fighter_dob[fid],
            )

        session = MagicMock()

        import hashlib

        # ── THE PARITY GATE: META cols 81-89 byte-identical ────────────────
        for fight in fight_records:
            fight_id = fight["fight_id"]
            event_date_val = fight["event_date"]
            # Replicate the deterministic swap from feature_matrix.assemble.
            swap = int(hashlib.md5(str(fight_id).encode()).hexdigest(), 16) % 2 == 0
            if swap:
                a_id = fight["fighter_b_id"]
                b_id = fight["fighter_a_id"]
            else:
                a_id = fight["fighter_a_id"]
                b_id = fight["fighter_b_id"]

            fa = _stub_fighter_orm(a_id)
            fb = _stub_fighter_orm(b_id)

            inf_row = inference_features.build(
                session,
                fa,
                fb,
                event_date_val,
                feature_set="v2.2",
                event_id=fight["event_id"],
            )[0]

            train_row_idx = next(
                i for i, f in enumerate(fight_records) if f["fight_id"] == fight_id
            )
            train_row = X_train[train_row_idx]

            for col_idx, col_name in (
                (81, "layoff_days_red"),
                (82, "layoff_days_blue"),
                (83, "age_at_fight_red"),
                (84, "age_at_fight_blue"),
                (85, "elo_overall_velocity_diff"),
                (86, "elo_striking_velocity_diff"),
                (87, "elo_grappling_velocity_diff"),
                (88, "division_finish_rate_shrunk"),
                (89, "reach_diff_normalized"),
            ):
                train_val = train_row[col_idx]
                inf_val = inf_row[col_idx]
                # NaN==NaN check
                if train_val != train_val and inf_val != inf_val:
                    continue
                assert train_val == _pytest.approx(inf_val, abs=1e-12), (
                    f"LIVE-03 META parity violation at fight {fight_id}, "
                    f"col {col_name} (idx {col_idx}): "
                    f"train={train_val} inference={inf_val}"
                )
