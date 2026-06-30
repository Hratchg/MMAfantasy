"""Tests for Phase 42 Plan 42-01 v2.5 TRAVEL feature-engineering close-out.

Phase 42 D-21 + D-22 (CONTEXT-locked):
    - TRAVEL-V25-01: `haversine_km` (kilometers, NaN-debut)
    - TRAVEL-V25-02: `compute_tz_shift_hours_clipped` (±12 clipped, NaN-debut)
    - `compute_travel_v25_features` composes both for one fighter's leg
    - `FEATURE_COLUMNS_V25_TRAVEL` = `FEATURE_COLUMNS_V22 + [travel_distance_km, tz_shift_hours]`
    - `FeatureMatrixAssembler.assemble(feature_set="v2.5-travel")` emits 92-col rows

Test groupings:
    - TestHaversineKm — great-circle distance correctness
    - TestTzShiftHoursClipped — DST-aware ±12-clipped tz shift
    - TestComputeTravelV25Features — composed per-fighter helper
    - TestFeatureColumnsV25Travel — config.py sibling constant
    - TestAssemblerV25TravelBranch — FeatureMatrixAssembler branch wiring

Sibling-not-replacement discipline: the existing v2.2 TRAVEL block
(FEATURE_COLUMNS_V22 indices 75-80, `compute_travel_features`,
`haversine_miles`) MUST stay byte-stable — regression-guarded by
`test_existing_compute_travel_features_unchanged` and
`test_v22_baseline_unchanged`.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

# ── TestHaversineKm ─────────────────────────────────────────────────────────


class TestHaversineKm:
    def test_t_mobile_to_msg_3612_km(self) -> None:
        """T-Mobile Arena (LV) → MSG (NYC) ≈ 3612 km ±1%."""
        from ufc_prediction.ml.features_v22.travel import haversine_km

        d = haversine_km(36.1027, -115.1761, 40.7505, -73.9934)
        # Standard reference distance ~3612 km — ±1% band = [3576, 3648].
        # Allow ±2% to absorb minor coord-precision variance.
        assert 3550 <= d <= 3680, f"T-Mobile→MSG distance {d} outside band"

    def test_sydney_to_london_17000_km(self) -> None:
        """Sydney → London ≈ 16983 km ±1%."""
        from ufc_prediction.ml.features_v22.travel import haversine_km

        d = haversine_km(-33.8688, 151.2093, 51.5074, -0.1278)
        assert 16800 <= d <= 17150, f"Sydney→London distance {d} outside band"

    def test_same_venue_returns_zero(self) -> None:
        """Same lat/lon pair → 0.0 exactly."""
        from ufc_prediction.ml.features_v22.travel import haversine_km

        assert haversine_km(36.1027, -115.1761, 36.1027, -115.1761) == 0.0

    def test_haversine_km_is_kilometers_not_miles(self) -> None:
        """haversine_km(A,B) / haversine_miles(A,B) ≈ 1.609 (km/mi)."""
        from ufc_prediction.ml.features_v22.travel import (
            haversine_km,
            haversine_miles,
        )

        km = haversine_km(36.1027, -115.1761, 40.7505, -73.9934)
        mi = haversine_miles(36.1027, -115.1761, 40.7505, -73.9934)
        ratio = km / mi
        # Standard km/mi = 1.609344; allow ±0.1% drift from earth-radius
        # constant differences (mean radius mi = 3958.7613, km = 6371.0088).
        assert 1.6075 <= ratio <= 1.6105, f"km/mi ratio {ratio} not ≈ 1.609"

    def test_haversine_km_symmetric(self) -> None:
        """haversine_km(A, B) == haversine_km(B, A) within float tol."""
        from ufc_prediction.ml.features_v22.travel import haversine_km

        a = haversine_km(36.1027, -115.1761, 40.7505, -73.9934)
        b = haversine_km(40.7505, -73.9934, 36.1027, -115.1761)
        assert a == pytest.approx(b, rel=1e-12)

    def test_earth_radius_km_constant(self) -> None:
        """EARTH_RADIUS_KM == 6371.0088 (IUGG mean radius)."""
        from ufc_prediction.ml.features_v22.travel import EARTH_RADIUS_KM

        assert EARTH_RADIUS_KM == 6371.0088


# ── TestTzShiftHoursClipped ─────────────────────────────────────────────────


class TestTzShiftHoursClipped:
    def test_nyc_to_la_winter_minus_3(self) -> None:
        """Winter: PST(-8) - EST(-5) = -3 (westward)."""
        from ufc_prediction.ml.features_v22.travel import (
            compute_tz_shift_hours_clipped,
        )

        # Both in winter (Jan 15) — EST and PST in effect.
        shift = compute_tz_shift_hours_clipped(
            "America/New_York",
            date(2026, 1, 15),
            "America/Los_Angeles",
            date(2026, 1, 15),
        )
        assert shift == pytest.approx(-3.0)

    def test_la_to_nyc_winter_plus_3(self) -> None:
        """Reverse direction: EST(-5) - PST(-8) = +3 (eastward)."""
        from ufc_prediction.ml.features_v22.travel import (
            compute_tz_shift_hours_clipped,
        )

        shift = compute_tz_shift_hours_clipped(
            "America/Los_Angeles",
            date(2026, 1, 15),
            "America/New_York",
            date(2026, 1, 15),
        )
        assert shift == pytest.approx(3.0)

    def test_dst_handled_correctly_nyc_to_la_summer(self) -> None:
        """Summer (DST): PDT(-7) - EDT(-4) = -3 (same magnitude as winter)."""
        from ufc_prediction.ml.features_v22.travel import (
            compute_tz_shift_hours_clipped,
        )

        shift = compute_tz_shift_hours_clipped(
            "America/New_York",
            date(2026, 7, 15),
            "America/Los_Angeles",
            date(2026, 7, 15),
        )
        assert shift == pytest.approx(-3.0)

    def test_cross_dst_boundary_la_to_la_zero_then_one_during_spring_forward(
        self,
    ) -> None:
        """Same venue across spring-forward → non-zero tz shift.

        2025-03-09 = US spring-forward (PST -8 → PDT -7). A fighter whose
        prior fight was on 2025-03-08 (PST -8) and current is 2025-03-10
        (PDT -7) should see curr - prior = -7 - -8 = +1 hour shift.
        """
        from ufc_prediction.ml.features_v22.travel import (
            compute_tz_shift_hours_clipped,
        )

        shift = compute_tz_shift_hours_clipped(
            "America/Los_Angeles",
            date(2025, 3, 8),
            "America/Los_Angeles",
            date(2025, 3, 10),
        )
        # PDT (-7) - PST (-8) = +1.0
        assert shift == pytest.approx(1.0)

    def test_clip_at_minus_12(self) -> None:
        """Auckland (+13 DST) → Honolulu (-10): raw -23h → clipped to -12.0."""
        from ufc_prediction.ml.features_v22.travel import (
            compute_tz_shift_hours_clipped,
        )

        # Jan 2026: Auckland is in NZDT (UTC+13); Honolulu HST (UTC-10).
        # raw = -10 - 13 = -23; clipped to -12.0.
        shift = compute_tz_shift_hours_clipped(
            "Pacific/Auckland",
            date(2026, 1, 15),
            "Pacific/Honolulu",
            date(2026, 1, 15),
        )
        assert shift == pytest.approx(-12.0)

    def test_clip_at_plus_12(self) -> None:
        """Honolulu (-10) → Auckland (+13 DST): raw +23h → clipped to +12.0."""
        from ufc_prediction.ml.features_v22.travel import (
            compute_tz_shift_hours_clipped,
        )

        shift = compute_tz_shift_hours_clipped(
            "Pacific/Honolulu",
            date(2026, 1, 15),
            "Pacific/Auckland",
            date(2026, 1, 15),
        )
        assert shift == pytest.approx(12.0)

    def test_debut_returns_nan_not_zero(self) -> None:
        """Debut (prior_tz_iana=None) → math.nan (NOT 0.0; v2.5 difference)."""
        from ufc_prediction.ml.features_v22.travel import (
            compute_tz_shift_hours_clipped,
        )

        result = compute_tz_shift_hours_clipped(
            None,
            None,
            "America/Los_Angeles",
            date(2026, 1, 15),
        )
        assert math.isnan(result), f"expected NaN, got {result}"

    def test_debut_prior_date_only_returns_nan(self) -> None:
        """prior_event_date=None (but tz given) also → NaN."""
        from ufc_prediction.ml.features_v22.travel import (
            compute_tz_shift_hours_clipped,
        )

        result = compute_tz_shift_hours_clipped(
            "America/New_York",
            None,
            "America/Los_Angeles",
            date(2026, 1, 15),
        )
        assert math.isnan(result)

    def test_nan_propagates_through_isnan_check(self) -> None:
        """math.isnan(result) is True for debut."""
        from ufc_prediction.ml.features_v22.travel import (
            compute_tz_shift_hours_clipped,
        )

        result = compute_tz_shift_hours_clipped(
            None,
            None,
            "America/Chicago",
            date(2026, 6, 1),
        )
        assert math.isnan(result) is True

    def test_tz_shift_cap_hours_constant(self) -> None:
        """TZ_SHIFT_CAP_HOURS == 12.0 (CONTEXT D-22 symmetric ±12 clip)."""
        from ufc_prediction.ml.features_v22.travel import TZ_SHIFT_CAP_HOURS

        assert TZ_SHIFT_CAP_HOURS == 12.0


# ── TestComputeTravelV25Features ────────────────────────────────────────────


class TestComputeTravelV25Features:
    def test_returns_dict_with_two_keys(self) -> None:
        """Returns {"travel_distance_km": float, "tz_shift_hours": float}."""
        from ufc_prediction.ml.features_v22.travel import (
            compute_travel_v25_features,
        )

        prior = {
            "lat": 36.1027,
            "lon": -115.1761,
            "timezone_iana": "America/Los_Angeles",
            "event_date": date(2025, 12, 1),
        }
        curr = {
            "lat": 40.7505,
            "lon": -73.9934,
            "timezone_iana": "America/New_York",
        }
        out = compute_travel_v25_features(prior, curr, date(2026, 1, 15))
        assert set(out.keys()) == {"travel_distance_km", "tz_shift_hours"}

    def test_debutant_returns_nan_pair(self) -> None:
        """prior_venue=None → both keys are math.nan."""
        from ufc_prediction.ml.features_v22.travel import (
            compute_travel_v25_features,
        )

        curr = {
            "lat": 40.7505,
            "lon": -73.9934,
            "timezone_iana": "America/New_York",
        }
        out = compute_travel_v25_features(None, curr, date(2026, 1, 15))
        assert math.isnan(out["travel_distance_km"])
        assert math.isnan(out["tz_shift_hours"])

    def test_veteran_returns_finite_pair(self) -> None:
        """Known prior + current → both finite floats."""
        from ufc_prediction.ml.features_v22.travel import (
            compute_travel_v25_features,
        )

        prior = {
            "lat": 36.1027,
            "lon": -115.1761,
            "timezone_iana": "America/Los_Angeles",
            "event_date": date(2025, 12, 1),
        }
        curr = {
            "lat": 40.7505,
            "lon": -73.9934,
            "timezone_iana": "America/New_York",
        }
        out = compute_travel_v25_features(prior, curr, date(2026, 1, 15))
        assert math.isfinite(out["travel_distance_km"])
        assert math.isfinite(out["tz_shift_hours"])
        # km > 0 because LA != NYC
        assert out["travel_distance_km"] > 0
        # tz_shift = NYC(-5 winter) - LA(-8 winter) = +3
        assert out["tz_shift_hours"] == pytest.approx(3.0)

    def test_existing_compute_travel_features_unchanged(self) -> None:
        """REGRESSION GUARD: v2.2 compute_travel_features still returns
        0.0-sentinel for debut (NOT NaN — that's the v2.5 difference)."""
        from ufc_prediction.ml.features_v22.travel import compute_travel_features

        curr = {
            "lat": 40.7505,
            "lon": -73.9934,
            "timezone_iana": "America/New_York",
        }
        out = compute_travel_features(None, curr, date(2026, 1, 15))
        # CRITICAL: v2.2 sentinel must remain 0.0; touching this would
        # invalidate meta_v2.joblib bytes (meta input space mutation).
        assert out["travel_distance_miles"] == 0.0
        assert out["tz_shift_signed"] == 0.0
        assert set(out.keys()) == {"travel_distance_miles", "tz_shift_signed"}


# ── TestFeatureColumnsV25Travel ─────────────────────────────────────────────


class TestFeatureColumnsV25Travel:
    def test_v25_travel_appends_to_v22(self) -> None:
        """First 90 entries of V25_TRAVEL byte-identical to V22."""
        from ufc_prediction.ml.config import (
            FEATURE_COLUMNS_V22,
            FEATURE_COLUMNS_V25_TRAVEL,
        )

        assert FEATURE_COLUMNS_V25_TRAVEL[:90] == FEATURE_COLUMNS_V22

    def test_v25_travel_length_92(self) -> None:
        """V25_TRAVEL = 90 v2.2 + 2 v2.5 = 92."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS_V25_TRAVEL

        assert len(FEATURE_COLUMNS_V25_TRAVEL) == 92

    def test_v25_travel_appends_km_then_hours(self) -> None:
        """Last 2 entries: ["travel_distance_km", "tz_shift_hours"]."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS_V25_TRAVEL

        assert FEATURE_COLUMNS_V25_TRAVEL[90:] == [
            "travel_distance_km",
            "tz_shift_hours",
        ]

    def test_v22_baseline_unchanged(self) -> None:
        """REGRESSION GUARD: FEATURE_COLUMNS_V22 length still 90 and the
        v2.2 TRAVEL block at indices 75-80 still holds the _miles_*_signed
        column names byte-identically. Touching this would invalidate
        meta_v2.joblib bytes (meta input space)."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS_V22

        assert len(FEATURE_COLUMNS_V22) == 90
        assert FEATURE_COLUMNS_V22[75:81] == [
            "travel_distance_miles_red",
            "travel_distance_miles_blue",
            "travel_distance_miles_diff",
            "tz_shift_red_signed",
            "tz_shift_blue_signed",
            "tz_shift_diff_signed",
        ]

    def test_get_feature_columns_dispatches_v25_travel(self) -> None:
        """get_feature_columns(feature_set='v2.5-travel') returns 92-col list."""
        from ufc_prediction.ml.config import (
            FEATURE_COLUMNS_V25_TRAVEL,
            get_feature_columns,
        )

        result = get_feature_columns(feature_set="v2.5-travel")
        assert result == list(FEATURE_COLUMNS_V25_TRAVEL)
        assert len(result) == 92

    def test_get_feature_columns_v22_path_unchanged(self) -> None:
        """REGRESSION GUARD: v2.2 dispatch path unchanged."""
        from ufc_prediction.ml.config import (
            FEATURE_COLUMNS_V22,
            get_feature_columns,
        )

        assert get_feature_columns(feature_set="v2.2") == list(FEATURE_COLUMNS_V22)
        assert len(get_feature_columns(feature_set="v2.2")) == 90


# ── TestAssemblerV25TravelBranch ────────────────────────────────────────────


def _build_synthetic_fixture():
    """3 fighters across 5 fights spanning 2 venues (T-Mobile + MSG).

    Fighter 1 (a_id=1) is a UFC debutant in fight 1, then has a prior
    venue for fights 2+. Fight 5 has a debutant blue fighter (a_id=4, new).

    Returns: (fight_records, elo_features, computed_features,
              fighter_physicals, division_medians).
    """
    # Venue dicts will live on each fight record via venue_lat/lon/tz keys.
    t_mobile = (36.1027, -115.1761, "America/Los_Angeles")
    msg = (40.7505, -73.9934, "America/New_York")

    fights = [
        # Fight 1: A=1 (debut), B=2 (debut), at T-Mobile (2025-06-01)
        {
            "fight_id": 1,
            "event_id": 100,
            "event_date": date(2025, 6, 1),
            "fighter_a_id": 1,
            "fighter_b_id": 2,
            "winner_id": 1,
            "weight_class": "Lightweight",
            "method": "Decision",
            "referee_id": None,
            "venue_lat": t_mobile[0],
            "venue_lon": t_mobile[1],
            "venue_timezone_iana": t_mobile[2],
        },
        # Fight 2: A=2, B=3 (debut), at MSG (2025-08-01) — A=2 has prior=T-Mobile
        {
            "fight_id": 2,
            "event_id": 200,
            "event_date": date(2025, 8, 1),
            "fighter_a_id": 2,
            "fighter_b_id": 3,
            "winner_id": 2,
            "weight_class": "Lightweight",
            "method": "Decision",
            "referee_id": None,
            "venue_lat": msg[0],
            "venue_lon": msg[1],
            "venue_timezone_iana": msg[2],
        },
        # Fight 3: A=1, B=3, at T-Mobile (2025-10-01) — both veterans
        {
            "fight_id": 3,
            "event_id": 300,
            "event_date": date(2025, 10, 1),
            "fighter_a_id": 1,
            "fighter_b_id": 3,
            "winner_id": 1,
            "weight_class": "Lightweight",
            "method": "Decision",
            "referee_id": None,
            "venue_lat": t_mobile[0],
            "venue_lon": t_mobile[1],
            "venue_timezone_iana": t_mobile[2],
        },
        # Fight 4: A=2, B=1, at MSG (2025-12-01) — both veterans
        {
            "fight_id": 4,
            "event_id": 400,
            "event_date": date(2025, 12, 1),
            "fighter_a_id": 2,
            "fighter_b_id": 1,
            "winner_id": 2,
            "weight_class": "Lightweight",
            "method": "Decision",
            "referee_id": None,
            "venue_lat": msg[0],
            "venue_lon": msg[1],
            "venue_timezone_iana": msg[2],
        },
        # Fight 5: A=3, B=4 (debut), at T-Mobile (2026-02-01) — B debut
        {
            "fight_id": 5,
            "event_id": 500,
            "event_date": date(2026, 2, 1),
            "fighter_a_id": 3,
            "fighter_b_id": 4,
            "winner_id": 3,
            "weight_class": "Lightweight",
            "method": "Decision",
            "referee_id": None,
            "venue_lat": t_mobile[0],
            "venue_lon": t_mobile[1],
            "venue_timezone_iana": t_mobile[2],
        },
    ]

    default_elo = {
        "elo_diff_overall": 0.0,
        "elo_diff_striking": 0.0,
        "elo_diff_grappling": 0.0,
        "elo_a_overall": 1500.0,
        "elo_a_striking": 1500.0,
        "elo_a_grappling": 1500.0,
        "elo_b_overall": 1500.0,
        "elo_b_striking": 1500.0,
        "elo_b_grappling": 1500.0,
    }
    elo_features = {(f["fighter_a_id"], f["fight_id"]): default_elo for f in fights}

    # Computed features default to 0 for all PERFORMANCE_FEATURE_KEYS.
    from ufc_prediction.ml.config import PERFORMANCE_FEATURE_KEYS

    default_perf = {k: 0.0 for k in PERFORMANCE_FEATURE_KEYS}
    computed_features = {(f["fighter_a_id"], f["fight_id"]): default_perf for f in fights}

    fighter_physicals = {
        fid: {
            "height_inches": 70.0,
            "reach_inches": 70.0,
            "leg_reach_inches": 40.0,
            "stance": "Orthodox",
            "date_of_birth": date(1990, 1, 1),
        }
        for fid in (1, 2, 3, 4)
    }
    division_medians = {
        "Lightweight": {
            "height_inches": 70.0,
            "reach_inches": 70.0,
            "leg_reach_inches": 40.0,
        }
    }
    return fights, elo_features, computed_features, fighter_physicals, division_medians


class TestAssemblerV25TravelBranch:
    def test_assemble_v25_travel_emits_92_cols(self) -> None:
        """assemble(feature_set='v2.5-travel') → shape[1] == 92."""
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        fights, elo, perf, phys, meds = _build_synthetic_fixture()
        asm = FeatureMatrixAssembler()
        X, y, dates = asm.assemble(
            fights,
            elo,
            perf,
            phys,
            meds,
            feature_set="v2.5-travel",
        )
        assert X.shape[1] == 92, f"expected 92 cols, got {X.shape[1]}"
        assert X.shape[0] == len(fights)

    def test_assemble_v25_travel_first_90_cols_equal_v22(self) -> None:
        """First 90 cols of v2.5-travel output = v2.2 output element-wise.

        Proves v2.5 path adds 2 cols without disturbing the v2.2 substrate.
        NaN-aware comparison (some v2.2 cols can be NaN for cold-start).
        """
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        fights, elo, perf, phys, meds = _build_synthetic_fixture()
        asm = FeatureMatrixAssembler()
        X22, _, _ = asm.assemble(
            fights,
            elo,
            perf,
            phys,
            meds,
            feature_set="v2.2",
        )
        X25, _, _ = asm.assemble(
            fights,
            elo,
            perf,
            phys,
            meds,
            feature_set="v2.5-travel",
        )
        # First 90 cols must be byte-identical (NaN-aware).
        assert X22.shape[1] == 90
        assert X25.shape[1] == 92
        # equal_nan=True so NaN==NaN counts as match.
        assert np.array_equal(X25[:, :90], X22, equal_nan=True), (
            "v2.5-travel first 90 cols drift vs v2.2"
        )

    def test_assemble_v25_travel_emits_nan_for_debutant(self) -> None:
        """Fight 1 (both debutants) → cols 90/91 are NaN."""
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        fights, elo, perf, phys, meds = _build_synthetic_fixture()
        asm = FeatureMatrixAssembler()
        X, _, _ = asm.assemble(
            fights,
            elo,
            perf,
            phys,
            meds,
            feature_set="v2.5-travel",
        )
        # Fight 1 (row 0): both A=1 and B=2 are debutants. red-blue = NaN-NaN = NaN.
        assert math.isnan(X[0, 90]), f"row 0 col 90 (km) should be NaN, got {X[0, 90]}"
        assert math.isnan(X[0, 91]), f"row 0 col 91 (hrs) should be NaN, got {X[0, 91]}"

    def test_assemble_v25_travel_emits_finite_for_veteran(self) -> None:
        """Fight 3 (both veterans with prior venues) → cols 90/91 finite."""
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        fights, elo, perf, phys, meds = _build_synthetic_fixture()
        asm = FeatureMatrixAssembler()
        X, _, _ = asm.assemble(
            fights,
            elo,
            perf,
            phys,
            meds,
            feature_set="v2.5-travel",
        )
        # Fight 3 (row 2): A=1 last fought at T-Mobile (fight 1);
        # B=3 last fought at MSG (fight 2). Current = T-Mobile.
        # A.km = haversine(T-Mobile, T-Mobile) = 0
        # B.km = haversine(MSG, T-Mobile) ≈ 3612
        # diff = 0 - 3612 = ~-3612 (negative; A had no travel, B traveled far)
        # A.tz = LA(-8) - LA(-8) = 0
        # B.tz = LA(-8) - NYC(-5) = -3
        # diff = 0 - (-3) = +3
        assert math.isfinite(X[2, 90])
        assert math.isfinite(X[2, 91])
        # Verify the differential sign + magnitude
        assert X[2, 90] == pytest.approx(-3612.0, abs=80.0), (
            f"fight 3 km diff = {X[2, 90]}, expected ~-3612"
        )
        assert X[2, 91] == pytest.approx(3.0), f"fight 3 hrs diff = {X[2, 91]}, expected ~+3"

    def test_assemble_v25_travel_uses_red_minus_blue_differential(self) -> None:
        """Red - Blue differential semantics verified explicitly.

        Fight 4 (original A=2, B=1) — assembler applies deterministic swap on
        fight_id=4 (md5("4")[0] is even), so post-swap a_id=1, b_id=2.

        a_id=1 prior=T-Mobile (from fight 1 + fight 3).
        b_id=2 prior=MSG (from fight 2). Current=MSG.

        a.km = haversine(T-Mobile, MSG) ≈ 3612
        b.km = haversine(MSG, MSG) = 0
        diff = 3612 - 0 = +3612 (a traveled far, b had no travel)

        a.tz = NYC(-5) - LA(-8) = +3
        b.tz = NYC(-5) - NYC(-5) = 0
        diff = +3 - 0 = +3
        """
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        fights, elo, perf, phys, meds = _build_synthetic_fixture()
        asm = FeatureMatrixAssembler()
        X, _, _ = asm.assemble(
            fights,
            elo,
            perf,
            phys,
            meds,
            feature_set="v2.5-travel",
        )
        # Fight 4 (row 3) — swap=True per md5("4"); post-swap a=1, b=2.
        assert X[3, 90] == pytest.approx(3612.0, abs=80.0), f"fight 4 km diff = {X[3, 90]}"
        assert X[3, 91] == pytest.approx(3.0), f"fight 4 hrs diff = {X[3, 91]}"

    def test_assemble_v22_path_unchanged_after_v25_landing(self) -> None:
        """REGRESSION GUARD: re-run v2.2 path; must emit 90-col rows
        and the new test_features_v22 + integration suites stay GREEN
        (those run separately in the CI gate; here we just spot-check
        the row shape + length-invariant assert)."""
        from ufc_prediction.ml.feature_matrix import (
            _EXPECTED_V22_NCOLS,
            FeatureMatrixAssembler,
        )

        fights, elo, perf, phys, meds = _build_synthetic_fixture()
        asm = FeatureMatrixAssembler()
        X, _, _ = asm.assemble(
            fights,
            elo,
            perf,
            phys,
            meds,
            feature_set="v2.2",
        )
        assert X.shape[1] == 90
        assert _EXPECTED_V22_NCOLS == 90
