"""Phase 65 FEAT-V261-02 — unit tests for ``features.referee_v2``.

Two waves of tests:

  Task 1 (helpers + RED body assertion):
    - ``derive_scoring_regime`` boundary (D-01 2017-01-01 split)
    - ``derive_event_country_bucket`` bucket map (D-02 6-bucket grouping)
    - ``EVENT_COUNTRY_BUCKETS`` constant exhaustiveness
    - Body still raises ``NotImplementedError`` (will flip GREEN in Task 2)

  Task 2 (body implementation — GREEN):
    - Zero-cohort fallback to global prior (Bayesian discipline)
    - Strict-pre-fight cutoff (Pitfall #5; ``event_date >= as_of_date`` excluded)
    - Small-cohort shrinkage toward prior
    - Closed-form Beta-binomial posterior match at k=50
    - k_shrink sensitivity (k=0 raw, k=50 default, k=200 stronger)
    - Pre-2017 / unified_post_2017 cohort isolation
    - UNKNOWN bucket allocation does not silently route through US
    - Strict same-day exclusion (event_date == as_of_date)
    - Return dtype + key contract (Python float, exactly 2 keys, ∈ [0, 1])
    - Method-string classification (no entries silently dropped)

Together: ≥10 tests, exceeding the must_haves ≥6 floor.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from ufc_prediction.features.referee_v2 import (
    EVENT_COUNTRY_BUCKETS,
    SCORING_REGIME_UNIFIED_START,
    RefereeV2Stratification,
    compute_ref_v2_features_shrunk,
    derive_event_country_bucket,
    derive_scoring_regime,
)

# ── Task 1: derive_scoring_regime (D-01) ───────────────────────────────────


class TestDeriveScoringRegime:
    """D-01: boundary at 2017-01-01; pre/post-2017 two-bucket regime."""

    def test_module_constant_pins_boundary(self) -> None:
        """The named constant prevents silent boundary drift (T-65-03)."""
        assert date(2017, 1, 1) == SCORING_REGIME_UNIFIED_START

    def test_derive_scoring_regime_boundary(self) -> None:
        """2016-12-31 is pre; 2017-01-01 is post (inclusive on post)."""
        assert derive_scoring_regime(date(2016, 12, 31)) == "pre_2017"
        assert derive_scoring_regime(date(2017, 1, 1)) == "unified_post_2017"
        assert derive_scoring_regime(date(2024, 6, 1)) == "unified_post_2017"

    def test_derive_scoring_regime_far_past(self) -> None:
        """UFC 1 (1993-11-12) is pre_2017."""
        assert derive_scoring_regime(date(1993, 11, 12)) == "pre_2017"

    def test_derive_scoring_regime_returns_only_two_labels(self) -> None:
        """The Phase 59 third tag (``alternative``) was DROPPED per D-01."""
        labels = {
            derive_scoring_regime(date(y, m, 15))
            for y in (2000, 2010, 2016, 2017, 2020, 2025)
            for m in (1, 6, 12)
        }
        assert labels == {"pre_2017", "unified_post_2017"}


# ── Task 1: derive_event_country_bucket (D-02) ─────────────────────────────


class TestDeriveEventCountryBucket:
    """D-02: 6-bucket grouping (US / BR / AU / ASIA-PACIFIC / EU-OTHER / UNKNOWN)."""

    def test_derive_event_country_bucket_us(self) -> None:
        assert derive_event_country_bucket("United States") == "US"

    def test_derive_event_country_bucket_br_au(self) -> None:
        assert derive_event_country_bucket("Brazil") == "BR"
        assert derive_event_country_bucket("Australia") == "AU"

    @pytest.mark.parametrize(
        "country",
        ["Japan", "China", "Singapore", "Philippines", "South Korea", "Thailand"],
    )
    def test_derive_event_country_bucket_asia_pacific(self, country: str) -> None:
        assert derive_event_country_bucket(country) == "ASIA-PACIFIC"

    @pytest.mark.parametrize(
        "country",
        [
            "England",
            "Scotland",
            "Ireland",
            "Germany",
            "France",
            "Sweden",
            "Spain",
            "Italy",
            "Poland",
            "Netherlands",
            "Czech Republic",
        ],
    )
    def test_derive_event_country_bucket_eu_other(self, country: str) -> None:
        assert derive_event_country_bucket(country) == "EU-OTHER"

    @pytest.mark.parametrize(
        "venue_country",
        [None, "", "Mongolia", "Antarctica", "Atlantis", "Wakanda"],
    )
    def test_derive_event_country_bucket_unknown(
        self,
        venue_country: str | None,
    ) -> None:
        """T-65-02: unmapped countries must fall through to ``UNKNOWN``,
        never silently route to US via fallthrough."""
        assert derive_event_country_bucket(venue_country) == "UNKNOWN"

    def test_event_country_buckets_constant_exhaustive(self) -> None:
        """The exported constant must match the bucket map exactly so the
        substrate builder (Plan 65-04) can iterate it canonically."""
        assert set(EVENT_COUNTRY_BUCKETS) == {
            "US",
            "BR",
            "AU",
            "ASIA-PACIFIC",
            "EU-OTHER",
            "UNKNOWN",
        }
        assert len(EVENT_COUNTRY_BUCKETS) == 6


# ── Task 2: body implementation (GREEN) ───────────────────────────────────
#
# All tests below pin specific behaviours of ``compute_ref_v2_features_shrunk``.
# Together with the Task 1 helper tests, this suite holds ≥10 GREEN tests
# (must_haves floor is ≥6).


# ── Shared test data ──────────────────────────────────────────────────────


AS_OF = date(2024, 6, 1)
US_POST = RefereeV2Stratification("US", "unified_post_2017")
US_PRE = RefereeV2Stratification("US", "pre_2017")


def _entry(d: date, method: str | None) -> dict[str, Any]:
    """Tiny helper to build a cohort_history entry."""
    return {"event_date": d, "method": method}


def _balanced_cohort(
    n_finishes: int,
    n_decisions: int,
    n_no_action: int = 0,
    *,
    before: date = AS_OF,
) -> list[dict[str, Any]]:
    """Build a list of cohort entries with the requested category counts.

    All entries are dated strictly before ``before`` (default AS_OF) so the
    strict-pre-fight filter does not exclude them.
    """
    entries: list[dict[str, Any]] = []
    base = date(before.year - 1, before.month, before.day)
    for _ in range(n_finishes):
        entries.append(_entry(base, "KO/TKO"))
    for _ in range(n_decisions):
        entries.append(_entry(base, "Decision - Unanimous"))
    for _ in range(n_no_action):
        entries.append(_entry(base, "No Contest"))
    return entries


# ── Bayesian fallback (zero cohort) ───────────────────────────────────────


class TestZeroCohortFallback:
    """Pitfall #2 / D-04: cohort absent OR effectively empty → prior fallback."""

    def test_body_returns_global_when_cohort_absent(self) -> None:
        """Empty ``cohort_history`` returns the prior on both columns."""
        out = compute_ref_v2_features_shrunk(
            event_country="US",
            scoring_regime="unified_post_2017",
            as_of_date=AS_OF,
            cohort_history={},
            global_finish_rate=0.45,
        )
        assert out["ref_v2_finish_rate_shrunk"] == pytest.approx(0.45)
        assert out["ref_v2_decision_rate_shrunk"] == pytest.approx(0.55)

    def test_body_returns_global_when_cohort_key_missing(self) -> None:
        """Non-empty history but the queried cohort key absent → fallback."""
        cohort_history = {US_PRE: _balanced_cohort(10, 10)}
        out = compute_ref_v2_features_shrunk(
            event_country="US",
            scoring_regime="unified_post_2017",  # not US_PRE
            as_of_date=AS_OF,
            cohort_history=cohort_history,
            global_finish_rate=0.40,
        )
        assert out["ref_v2_finish_rate_shrunk"] == pytest.approx(0.40)
        assert out["ref_v2_decision_rate_shrunk"] == pytest.approx(0.60)

    def test_body_returns_global_when_only_future_entries(self) -> None:
        """Entries with ``event_date >= as_of_date`` are excluded; n=0 → fallback."""
        future_only = [
            _entry(AS_OF, "KO/TKO"),  # same-day excluded (strict >= cutoff)
            _entry(date(2024, 7, 1), "Decision - Unanimous"),
            _entry(date(2025, 1, 1), "Submission"),
        ]
        out = compute_ref_v2_features_shrunk(
            event_country="US",
            scoring_regime="unified_post_2017",
            as_of_date=AS_OF,
            cohort_history={US_POST: future_only},
            global_finish_rate=0.50,
        )
        assert out["ref_v2_finish_rate_shrunk"] == pytest.approx(0.50)
        assert out["ref_v2_decision_rate_shrunk"] == pytest.approx(0.50)


# ── Strict pre-fight discipline (Pitfall #5) ──────────────────────────────


class TestStrictPreFightDiscipline:
    """Pitfall #5: ``event_date >= as_of_date`` entries excluded."""

    def test_body_strict_pre_fight_excludes_same_day(self) -> None:
        """An entry dated exactly ``as_of_date`` is excluded (strict ``>=``).

        Mirrors features_v22/ref.py line 145: same-day fights are excluded
        to prevent the current event's own outcome from leaking into its
        own pre-fight feature row.
        """
        cohort_history = {
            US_POST: [
                _entry(AS_OF, "KO/TKO"),  # same-day — must be excluded
            ],
        }
        out = compute_ref_v2_features_shrunk(
            event_country="US",
            scoring_regime="unified_post_2017",
            as_of_date=AS_OF,
            cohort_history=cohort_history,
            global_finish_rate=0.45,
        )
        # n effective = 0 → fallback to prior on both columns.
        assert out["ref_v2_finish_rate_shrunk"] == pytest.approx(0.45)
        assert out["ref_v2_decision_rate_shrunk"] == pytest.approx(0.55)

    def test_body_mixed_past_and_future_only_past_counted(self) -> None:
        """Past entries contribute; future entries are silently filtered."""
        cohort_history = {
            US_POST: [
                _entry(date(2023, 1, 1), "KO/TKO"),  # counted (finish)
                _entry(date(2023, 6, 1), "Decision - Unanimous"),  # counted (decision)
                _entry(AS_OF, "KO/TKO"),  # excluded (same-day)
                _entry(date(2024, 8, 1), "Decision - Split"),  # excluded (future)
            ],
        }
        out = compute_ref_v2_features_shrunk(
            event_country="US",
            scoring_regime="unified_post_2017",
            as_of_date=AS_OF,
            cohort_history=cohort_history,
            global_finish_rate=0.45,
            k_shrink=0.0,  # no shrinkage → raw rates from past entries only
        )
        # 1 finish / 1 decision out of n=2 past entries.
        assert out["ref_v2_finish_rate_shrunk"] == pytest.approx(0.5)
        assert out["ref_v2_decision_rate_shrunk"] == pytest.approx(0.5)


# ── Closed-form Beta-binomial math ────────────────────────────────────────


class TestBetaBinomialClosedForm:
    """D-04: ``(successes + k * prior) / (n + k)`` closed-form."""

    def test_body_matches_closed_form_at_k50(self) -> None:
        """100 fights, 40 finishes, 55 decisions, global=0.45, k=50 → exact."""
        cohort_history = {
            US_POST: _balanced_cohort(40, 55, n_no_action=5),
        }
        out = compute_ref_v2_features_shrunk(
            event_country="US",
            scoring_regime="unified_post_2017",
            as_of_date=AS_OF,
            cohort_history=cohort_history,
            global_finish_rate=0.45,
            k_shrink=50.0,
        )
        # finish prior = 0.45; decision prior = 1 - 0.45 = 0.55
        expected_finish = (40 + 50 * 0.45) / (100 + 50)
        expected_decision = (55 + 50 * 0.55) / (100 + 50)
        assert out["ref_v2_finish_rate_shrunk"] == pytest.approx(expected_finish)
        assert out["ref_v2_decision_rate_shrunk"] == pytest.approx(expected_decision)

    def test_body_shrinks_small_cohort_toward_prior(self) -> None:
        """Tiny cohort (n=2) gets pulled toward the global prior."""
        cohort_history = {
            US_POST: _balanced_cohort(1, 1),  # 1 finish, 1 decision → raw = 0.5
        }
        out = compute_ref_v2_features_shrunk(
            event_country="US",
            scoring_regime="unified_post_2017",
            as_of_date=AS_OF,
            cohort_history=cohort_history,
            global_finish_rate=0.40,
            k_shrink=50.0,
        )
        # raw finish = 0.5; prior = 0.4; shrunk lies strictly between.
        assert 0.40 < out["ref_v2_finish_rate_shrunk"] < 0.50
        # raw decision = 0.5; prior = 0.6; shrunk lies strictly between.
        assert 0.50 < out["ref_v2_decision_rate_shrunk"] < 0.60

    def test_body_k_shrink_sensitivity(self) -> None:
        """Larger k → stronger pull toward prior."""
        cohort_history = {
            US_POST: _balanced_cohort(8, 2),  # 8 finishes / 2 decisions = 0.8 raw
        }
        global_finish_rate = 0.45  # prior

        def _finish_at(k: float) -> float:
            return compute_ref_v2_features_shrunk(
                event_country="US",
                scoring_regime="unified_post_2017",
                as_of_date=AS_OF,
                cohort_history=cohort_history,
                global_finish_rate=global_finish_rate,
                k_shrink=k,
            )["ref_v2_finish_rate_shrunk"]

        finish_k0 = _finish_at(0.0)
        finish_k50 = _finish_at(50.0)
        finish_k200 = _finish_at(200.0)

        # k=0 is the raw rate (no shrinkage at all).
        assert finish_k0 == pytest.approx(0.8)
        # Distance from prior shrinks monotonically as k grows.
        assert (
            abs(finish_k200 - global_finish_rate)
            < abs(
                finish_k50 - global_finish_rate,
            )
            < abs(finish_k0 - global_finish_rate)
        )

    def test_body_no_action_dilutes_via_denominator(self) -> None:
        """no_action entries inflate ``n`` (denominator) but neither numerator."""
        # 50 fights total: 10 finishes, 10 decisions, 30 no_action.
        cohort_history = {
            US_POST: _balanced_cohort(10, 10, n_no_action=30),
        }
        out = compute_ref_v2_features_shrunk(
            event_country="US",
            scoring_regime="unified_post_2017",
            as_of_date=AS_OF,
            cohort_history=cohort_history,
            global_finish_rate=0.40,
            k_shrink=0.0,  # isolate the no_action effect
        )
        # Raw rates with n=50 are 10/50 = 0.2, NOT 10/(10+10) = 0.5.
        assert out["ref_v2_finish_rate_shrunk"] == pytest.approx(0.2)
        assert out["ref_v2_decision_rate_shrunk"] == pytest.approx(0.2)


# ── Cohort isolation (pre / post 2017; UNKNOWN bucket) ───────────────────


class TestCohortIsolation:
    """D-01 + D-02: cohorts must NOT cross-pollinate."""

    def test_body_pre_post_2017_cohorts_isolated(self) -> None:
        """pre_2017 finish-heavy cohort must not bleed into the post_2017 query."""
        cohort_history = {
            US_PRE: _balanced_cohort(100, 0),  # 100 finishes / 0 decisions
            US_POST: _balanced_cohort(0, 100),  # 0 finishes / 100 decisions
        }
        global_finish_rate = 0.45

        post = compute_ref_v2_features_shrunk(
            event_country="US",
            scoring_regime="unified_post_2017",
            as_of_date=AS_OF,
            cohort_history=cohort_history,
            global_finish_rate=global_finish_rate,
        )
        pre = compute_ref_v2_features_shrunk(
            event_country="US",
            scoring_regime="pre_2017",
            as_of_date=AS_OF,
            cohort_history=cohort_history,
            global_finish_rate=global_finish_rate,
        )

        # post cohort is decision-dominated → shrunk finish below prior
        assert post["ref_v2_finish_rate_shrunk"] < global_finish_rate
        # pre cohort is finish-dominated → shrunk finish well above prior
        assert pre["ref_v2_finish_rate_shrunk"] > global_finish_rate
        # And they must NOT be equal (no silent cross-pollination)
        assert post["ref_v2_finish_rate_shrunk"] != pytest.approx(
            pre["ref_v2_finish_rate_shrunk"],
        )

    def test_body_unknown_bucket_allocation(self) -> None:
        """``("UNKNOWN", "unified_post_2017")`` does not silently route to US."""
        # Populate US cohort but query UNKNOWN — should NOT see US data.
        cohort_history = {US_POST: _balanced_cohort(100, 0)}  # 100 finishes
        global_finish_rate = 0.40

        out_unknown = compute_ref_v2_features_shrunk(
            event_country="UNKNOWN",
            scoring_regime="unified_post_2017",
            as_of_date=AS_OF,
            cohort_history=cohort_history,
            global_finish_rate=global_finish_rate,
        )
        # UNKNOWN cohort key is absent → must fall back to prior, NOT
        # leak the US 100-finish signal.
        assert out_unknown["ref_v2_finish_rate_shrunk"] == pytest.approx(
            global_finish_rate,
        )

        # When the UNKNOWN cohort IS populated, the query respects it.
        unknown_key = RefereeV2Stratification("UNKNOWN", "unified_post_2017")
        cohort_history[unknown_key] = _balanced_cohort(50, 0)  # 50 finishes
        out_unknown_pop = compute_ref_v2_features_shrunk(
            event_country="UNKNOWN",
            scoring_regime="unified_post_2017",
            as_of_date=AS_OF,
            cohort_history=cohort_history,
            global_finish_rate=global_finish_rate,
        )
        # finish-heavy → shrunk above prior, distinct from the fallback.
        assert out_unknown_pop["ref_v2_finish_rate_shrunk"] > global_finish_rate


# ── Return-shape contract ─────────────────────────────────────────────────


class TestReturnContract:
    """D-04: exactly two ``float`` keys, both in [0, 1]."""

    def test_body_return_dtype_and_keys(self) -> None:
        """Both populated and fallback paths return Python float in [0, 1]."""
        # Populated path
        out_pop = compute_ref_v2_features_shrunk(
            event_country="US",
            scoring_regime="unified_post_2017",
            as_of_date=AS_OF,
            cohort_history={US_POST: _balanced_cohort(40, 55, n_no_action=5)},
            global_finish_rate=0.45,
        )
        # Fallback path
        out_fb = compute_ref_v2_features_shrunk(
            event_country="UNKNOWN",
            scoring_regime="pre_2017",
            as_of_date=AS_OF,
            cohort_history={},
            global_finish_rate=0.45,
        )

        for out in (out_pop, out_fb):
            assert set(out.keys()) == {
                "ref_v2_finish_rate_shrunk",
                "ref_v2_decision_rate_shrunk",
            }
            for value in out.values():
                # Must be plain Python float, not numpy scalar.
                assert isinstance(value, float)
                assert type(value) is float
                assert 0.0 <= value <= 1.0

    def test_body_classifies_methods_via_classify_outcome(self) -> None:
        """Every entry in the cohort contributes to ``n`` (no silent drops).

        Mix one of each method-string category; with k_shrink=0 the raw rate
        denominator must equal the entry count, proving no entry was silently
        dropped from ``n``.
        """
        cohort_history = {
            US_POST: [
                _entry(date(2023, 1, 1), "KO/TKO"),  # finish
                _entry(date(2023, 2, 1), "TKO"),  # finish
                _entry(date(2023, 3, 1), "Submission"),  # finish
                _entry(date(2023, 4, 1), "SUB"),  # finish
                _entry(date(2023, 5, 1), "Decision"),  # decision
                _entry(date(2023, 6, 1), "Decision - Unanimous"),  # decision
                _entry(date(2023, 7, 1), "Decision - Split"),  # decision
                _entry(date(2023, 8, 1), "Decision - Majority"),  # decision
                _entry(date(2023, 9, 1), "No Contest"),  # no_action
                _entry(date(2023, 10, 1), None),  # no_action
            ],
        }
        out = compute_ref_v2_features_shrunk(
            event_country="US",
            scoring_regime="unified_post_2017",
            as_of_date=AS_OF,
            cohort_history=cohort_history,
            global_finish_rate=0.40,
            k_shrink=0.0,
        )
        # 4 finishes / 10 total = 0.4; 4 decisions / 10 total = 0.4
        # (the 2 no_action entries inflate the denominator).
        assert out["ref_v2_finish_rate_shrunk"] == pytest.approx(0.4)
        assert out["ref_v2_decision_rate_shrunk"] == pytest.approx(0.4)
