"""Phase 59 FEAT-V26-01 / Phase 65 FEAT-V261-02 — REF feature v2.

Redesign per Phase 26 + Phase 32 evidence: per-referee shrunk rates
failed to compose above CALIB. v2.6 hypothesis: stratifying by event
country / scoring regime captures the systematic effect (judging
standards differ by jurisdiction) without the per-referee sample-size
fragility.

Phase 59 shipped scaffold only (signature + dataclass). Phase 65
(FEAT-V261-02, D-01 + D-02 LOCKED 2026-06-05) fills the body and adds
the two helpers ``derive_scoring_regime`` + ``derive_event_country_bucket``
so downstream substrate / meta callers do not re-declare the bucket map.

Math reference: closed-form Beta-binomial posterior mean
    posterior_mean = (successes + k_shrink * global_rate) / (n + k_shrink)
mirrors ``features_v22/ref.py`` Phase 26 D-02 rule-of-thumb
``k_shrink = 50.0``. Only the cohort grouping changes — per-(event_country,
scoring_regime) instead of per-referee.

See ``results/ref_redesign_spike_v26.md`` for the design rationale and
``.planning/phases/65-.../65-CONTEXT.md`` for D-01..D-10 decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from ufc_prediction.ml.features_v22.ref import classify_outcome


# ── Module-level constants (D-01 + D-02 LOCKED) ────────────────────────────

#: Date on which the unified MMA rules took effect (Phase 65 D-01 boundary).
#: Events with ``event_date < SCORING_REGIME_UNIFIED_START`` map to the
#: ``"pre_2017"`` regime; events on-or-after map to ``"unified_post_2017"``.
#: The Phase 59 scaffold also mentioned a third ``"alternative"`` tag — this
#: was DROPPED in Phase 65 D-01 (no data source supports it; two-bucket
#: regime is signal-realistic without inventing data).
SCORING_REGIME_UNIFIED_START: date = date(2017, 1, 1)

#: Canonical 6-bucket event-country grouping per Phase 65 D-02. Exported so
#: downstream substrate builders (Plan 65-04) iterate the canonical list
#: without re-declaring it. ``"UNKNOWN"`` is the catch-all for ``None`` /
#: empty / unmapped countries.
EVENT_COUNTRY_BUCKETS: tuple[str, ...] = (
    "US",
    "BR",
    "AU",
    "ASIA-PACIFIC",
    "EU-OTHER",
    "UNKNOWN",
)

# Closed sets backing ``derive_event_country_bucket`` (Phase 65 D-02). Module-
# level frozensets to avoid per-call set construction. If the live Venue.country
# distribution surfaces additional European or Asia-Pacific countries that should
# bucket together, extend these sets — but keep the public bucket list at the
# 6 strings above so downstream callers do not break.
_ASIA_PACIFIC_COUNTRIES: frozenset[str] = frozenset({
    "Japan",
    "China",
    "Singapore",
    "Philippines",
    "South Korea",
    "Thailand",
})

_EU_OTHER_COUNTRIES: frozenset[str] = frozenset({
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
})


# ── Stratification cohort key ──────────────────────────────────────────────


@dataclass(frozen=True)
class RefereeV2Stratification:
    """One stratification cohort for the v2.6 REF redesign.

    Two-axis stratification:
      - ``event_country``: aggregated jurisdiction bucket from
        :func:`derive_event_country_bucket` — one of
        ``"US"``, ``"BR"``, ``"AU"``, ``"ASIA-PACIFIC"``, ``"EU-OTHER"``,
        ``"UNKNOWN"`` (Phase 65 D-02).
      - ``scoring_regime``: scoring-rules tag from
        :func:`derive_scoring_regime` — one of ``"pre_2017"`` or
        ``"unified_post_2017"`` (Phase 65 D-01).

    The cartesian product (``event_country`` × ``scoring_regime``) is the
    cohort space. Phase 65 implementation populates per-cohort shrunk
    finish-rate + decision-rate metrics on a Bayesian Beta-binomial
    backbone (mirrors ``features_v22/ref.py`` Phase 26 helper math).
    """

    event_country: str
    scoring_regime: str


# ── Helpers (D-01 + D-02) ──────────────────────────────────────────────────


def derive_scoring_regime(event_date: date) -> str:
    """Return the scoring-regime label for an event date (Phase 65 D-01).

    Two-bucket regime derived from ``event.event_date`` (the schema has no
    ``scoring_regime`` column; D-01 RESOLVED this by deriving the label
    from the universal event-date column).

    - ``event_date < date(2017, 1, 1)`` → ``"pre_2017"``
    - ``event_date >= date(2017, 1, 1)`` → ``"unified_post_2017"``

    The 2017-01-01 boundary corresponds to the unified MMA rules transition;
    pre/post is the most meaningful regime split that does not require new
    data. The Phase 59 scaffold mentioned a third ``"alternative"`` tag —
    Phase 65 D-01 DROPPED it (no data source supports it).
    """
    if event_date < SCORING_REGIME_UNIFIED_START:
        return "pre_2017"
    return "unified_post_2017"


def derive_event_country_bucket(venue_country: str | None) -> str:
    """Return the event-country bucket label for a Venue.country (Phase 65 D-02).

    The join path that produces ``venue_country`` is
    ``event.venue_id (NULLable) → venue.country (NOT NULL)``. The NULLable
    FK means callers WILL hand us ``None`` for events with no venue, so
    ``None`` and empty-string inputs are tolerated and return ``"UNKNOWN"``.

    Bucket map (D-02 LOCKED):
      - ``"United States"`` → ``"US"``
      - ``"Brazil"`` → ``"BR"``
      - ``"Australia"`` → ``"AU"``
      - ``{"Japan", "China", "Singapore", "Philippines",
            "South Korea", "Thailand"}`` → ``"ASIA-PACIFIC"``
      - ``{"England", "Scotland", "Ireland", "Germany", "France",
            "Sweden", "Spain", "Italy", "Poland", "Netherlands",
            "Czech Republic"}`` → ``"EU-OTHER"``
      - ``None``, empty string, or anything else → ``"UNKNOWN"``

    Bucketing balances cohort size (Beta-binomial shrinkage prior wants
    N≥30 typically) against jurisdictional homogeneity. The 6-bucket
    grouping is exported via :data:`EVENT_COUNTRY_BUCKETS` for the
    Plan 65-04 substrate builder.
    """
    if venue_country is None or venue_country == "":
        return "UNKNOWN"
    if venue_country == "United States":
        return "US"
    if venue_country == "Brazil":
        return "BR"
    if venue_country == "Australia":
        return "AU"
    if venue_country in _ASIA_PACIFIC_COUNTRIES:
        return "ASIA-PACIFIC"
    if venue_country in _EU_OTHER_COUNTRIES:
        return "EU-OTHER"
    return "UNKNOWN"


# ── Per-cohort shrunk-rate compute (Phase 65 D-04 deliverable) ────────────


def compute_ref_v2_features_shrunk(
    event_country: str,
    scoring_regime: str,
    as_of_date: date,
    *,
    cohort_history: dict[RefereeV2Stratification, list[dict[str, Any]]],
    global_finish_rate: float,
    k_shrink: float = 50.0,
) -> dict[str, float]:
    """Compute REF v2 features for one (cohort, as_of_date) pair.

    Phase 65 D-04 deliverable: two-column output, Beta-binomial closed-form
    posterior-mean shrinkage on ``(event_country, scoring_regime)`` cohorts.
    Mirrors ``features_v22.ref.compute_ref_rates_shrunk`` math but groups
    by jurisdiction-regime instead of per-referee.

    Args:
        event_country: Bucket label from :func:`derive_event_country_bucket`.
        scoring_regime: Regime label from :func:`derive_scoring_regime`.
        as_of_date: Strict pre-fight as-of-date discipline (Pitfall #5).
            Entries with ``event_date >= as_of_date`` are excluded — even
            same-day fights, to avoid bleeding the current event's own
            outcome (mirrors ``features_v22.ref.py`` line 145).
        cohort_history: ``{RefereeV2Stratification(...): [
            {"event_date": date, "method": str | None}, ...]}``. Operator
            contract: ideally pre-filtered to ``event_date < as_of_date``,
            but this function defensively re-filters (Pitfall #5).
        global_finish_rate: Corpus-level finish rate used as the
            Beta-binomial prior for the finish column. The decision prior
            is derived as ``decision_prior = 1.0 - global_finish_rate``
            (D-04 spec: only two output columns; the "no_action" mass is
            folded into the shrinkage denominator ``n``, shrinking BOTH
            rates toward their priors at the same speed).
        k_shrink: Shrinkage strength. Default ``50.0`` per Phase 26 D-02
            rule of thumb. ``k_shrink == 0.0`` returns the raw observed
            rates (no shrinkage). CV-tuning of ``k_shrink`` is BANNED
            per Phase 26 D-02.

    Returns:
        ``{"ref_v2_finish_rate_shrunk": float, "ref_v2_decision_rate_shrunk": float}``.
        Both values are plain Python ``float`` in ``[0.0, 1.0]`` (no numpy
        scalars). Method classification (finish / decision / no_action) uses
        :func:`ufc_prediction.ml.features_v22.ref.classify_outcome` so the
        method-string semantics match the Phase 26 baseline byte-for-byte.

    Bayesian fallback discipline (Pitfall #2 / Pitfall #5): when the cohort
    is absent from ``cohort_history`` OR all entries are filtered by the
    strict-pre-fight cutoff, ``n == 0`` and the closed-form formula returns
    exactly ``(global_finish_rate, 1 - global_finish_rate)`` — no
    zero-division.
    """
    cohort_key = RefereeV2Stratification(
        event_country=event_country,
        scoring_regime=scoring_regime,
    )
    decision_prior = 1.0 - global_finish_rate

    cohort_entries = cohort_history.get(cohort_key, [])

    finish_count = 0
    decision_count = 0
    n = 0
    for past in cohort_entries:
        # Strict less-than cutoff (Pitfall #5 — same-day fights excluded
        # to avoid bleeding the current event's own outcome). Mirrors
        # features_v22/ref.py line 145.
        if past["event_date"] >= as_of_date:
            continue
        cat = classify_outcome(past.get("method"))
        if cat == "finish":
            finish_count += 1
        elif cat == "decision":
            decision_count += 1
        # "no_action" still increments n (denominator inflation), so both
        # rates shrink toward their priors at the same speed.
        n += 1

    denom = n + k_shrink
    if denom == 0:
        # k_shrink == 0 AND n == 0 — raw rate undefined; fall back to prior
        # (Bayesian discipline; Pitfall #2 no zero-division).
        return {
            "ref_v2_finish_rate_shrunk": float(global_finish_rate),
            "ref_v2_decision_rate_shrunk": float(decision_prior),
        }

    finish_shrunk = (finish_count + k_shrink * global_finish_rate) / denom
    decision_shrunk = (decision_count + k_shrink * decision_prior) / denom

    return {
        "ref_v2_finish_rate_shrunk": float(finish_shrunk),
        "ref_v2_decision_rate_shrunk": float(decision_shrunk),
    }


# ── ML spike + gate verification hook (Phase 65 follow-on plans) ─────────
#
# Phase 65 plan sequence:
#   1. Plan 65-01 (this file): implement compute_ref_v2_features_shrunk +
#      derive_scoring_regime + derive_event_country_bucket helpers + tests
#   2. Plan 65-02: scripts/retrain_xgb_v2_refv2.py — xgb retrain with REF v2
#      cols appended to v2.2 90-col feature space → xgb_v2_refv2.joblib
#      (sibling; canonical xgb_v2.joblib UNTOUCHED) + OOF predictions parquet
#   3. Plan 65-03: scripts/train_meta_v2_refv2.py — logistic-regression meta
#      with col[0] = xgb_v2_refv2_oof → meta_v2_refv2.joblib sibling
#   4. Plan 65-04: scripts/build_ref_substrate_v261.py — 3-slice substrate
#      parquet with candidate-aligned col[0]
#   5. Plan 65-05: ufc gate verify → results/ref_redesign_gate_v261.{md,json}
#      → path-conditional close (Path A / Path B / confound_block)
