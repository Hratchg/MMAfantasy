"""Predict-time twin of feature_matrix.py (LIVE-02, closes 44/72 NaN-pad gap).

Reads latest per-fighter ``ComputedFeature`` + ``EloSnapshot`` snapshots from
the DB and builds a ``(1, len(FEATURE_COLUMNS))``-shaped feature vector for
a single matchup.

Per CONTEXT.md D-12: this module REPLACES the NaN-pad block at
``predictor.py:289-352``. The NaN-pad block is deleted in 16-02 Task 4.

Per Gotcha 2 / Pattern Map: the 5-feature Phase 15.1 odds block at
``feature_matrix.py:560-625`` is replicated here column-for-column.

Per Pattern D: NaN — never 0.0 — for missing odds. XGBoost handles native
via sparsity-aware split finding (D-04(P14)).

Per Pitfall #12: column ordering is locked by ``FEATURE_COLUMNS``; a strict
``unknown_keys`` guard prevents silent positional drift.

The five DB-reading helpers (``_get_latest_elo``, ``_get_latest_computed_features``,
``_get_fighter_physical``, ``_get_cached_odds``) are module-level so tests can
monkey-patch them without spinning up Postgres.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ufc_prediction.dedup.source_priority import prefer_canonical
from ufc_prediction.ml.config import (
    FEATURE_COLUMNS_V22,
    PERFORMANCE_FEATURE_KEYS,
    encode_stance_matchup,
    get_feature_columns,
)
from ufc_prediction.ml.features_v22.meta import (
    age_at_fight,
    division_finish_rate_shrunk,
    elo_velocity,
    layoff_days,
    reach_diff_normalized,
)
from ufc_prediction.ml.features_v22.ref import (
    classify_outcome,
    compute_ref_rates_shrunk,
)
from ufc_prediction.ml.features_v22.travel import (
    compute_travel_features,
)
from ufc_prediction.models.computed_feature import ComputedFeature
from ufc_prediction.models.elo_snapshot import EloSnapshot
from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fight_odds import FightOdds
from ufc_prediction.models.fighter import Fighter
from ufc_prediction.models.venue import Venue
from ufc_prediction.scraper.bfo_math import (
    InvalidMoneylineError,
    devig_closing_range,
    devig_proportional,
)

if TYPE_CHECKING:
    from ufc_prediction.scraper.bfo_live import MatchupOdds

logger = logging.getLogger(__name__)


# ── DB-reading helpers ──────────────────────────────────────────────────────


def _get_latest_elo(session: Session, fighter_id: int, elo_type: str) -> float:
    """Read latest ``elo_after_shrinkage`` for a fighter (lifted from
    predictor.py:_get_latest_elo).

    Returns 1500.0 fallback when no snapshots exist (debutant case).
    """
    stmt = (
        select(EloSnapshot.elo_after_shrinkage)
        .where(EloSnapshot.fighter_id == fighter_id)
        .where(EloSnapshot.elo_type == elo_type)
        .order_by(EloSnapshot.fight_date.desc())
        .limit(1)
    )
    result = session.scalar(stmt)
    return float(result) if result is not None else 1500.0


def _get_latest_computed_features(
    session: Session,
    fighter_id: int,
) -> dict[str, float | None]:
    """Read latest ``ComputedFeature.features`` JSON (lifted from
    predictor.py:_get_latest_computed_features).

    Returns empty dict on miss. Caller falls through to NaN per
    Pattern D / Pitfall #3.

    Also lifts the 3 NET-* keys (``pagerank``, ``sos_2hop``,
    ``is_debutant_in_graph``) from the JSONB so ``_populate_network``
    can compute the 3 ``_diff`` columns. NaN/None propagates per
    Pattern D for debutants (D-06).
    """
    stmt = (
        select(ComputedFeature.features)
        .where(ComputedFeature.fighter_id == fighter_id)
        .order_by(ComputedFeature.as_of_date.desc())
        .limit(1)
    )
    features_json = session.scalar(stmt)
    if features_json is None:
        return {}
    extracted: dict[str, float | None] = {
        feat: features_json.get(feat) for feat in PERFORMANCE_FEATURE_KEYS
    }
    # NET-* keys (Phase 16-03) — written by Pass 4 of FeatureComputer.
    for net_key in ("pagerank", "sos_2hop", "is_debutant_in_graph"):
        extracted[net_key] = features_json.get(net_key)
    return extracted


def _get_cached_odds(
    session: Session,
    fa_id: int,
    fb_id: int,
    event_date: date,
) -> tuple[dict | None, dict | None]:
    """Look up cached fight_odds rows for ``(A, B, event_date)``.

    Returns ``(odds_a, odds_b)`` as plain dicts, each with at minimum
    ``opening_implied_prob`` and ``closing_implied_prob`` keys (matching
    the cache shape ``feature_matrix.py:569`` reads). Returns ``(None, None)``
    on any lookup failure — caller treats as cache miss.
    """
    try:
        stmt = (
            select(Fight.id)
            .join(Event, Fight.event_id == Event.id)
            .where(
                ((Fight.fighter_a_id == fa_id) & (Fight.fighter_b_id == fb_id))
                | ((Fight.fighter_a_id == fb_id) & (Fight.fighter_b_id == fa_id))
            )
            .where(Event.date == event_date)
        )
        fight_id = session.scalar(stmt)
        if fight_id is None:
            return None, None

        rows = list(
            session.execute(select(FightOdds).where(FightOdds.fight_id == fight_id)).scalars().all()
        )
        if not rows:
            return None, None

        odds_a_row = next((r for r in rows if r.fighter_id == fa_id), None)
        odds_b_row = next((r for r in rows if r.fighter_id == fb_id), None)
        if odds_a_row is None or odds_b_row is None:
            return None, None

        return (
            {
                "opening_implied_prob": odds_a_row.opening_implied_prob,
                "closing_implied_prob": odds_a_row.closing_implied_prob,
            },
            {
                "opening_implied_prob": odds_b_row.opening_implied_prob,
                "closing_implied_prob": odds_b_row.closing_implied_prob,
            },
        )
    except Exception as exc:
        logger.warning("inference_features cache lookup failed: %s", exc)
        return None, None


def _resolve_fighter_id(session: Session, name: str) -> int | None:
    """Resolve fighter name → single canonical id. Used only by ``build``
    when the caller passes Fighter ORM rows that haven't been written yet.

    Returns None when no row matches; caller treats as cache miss.
    """
    rows = list(session.execute(select(Fighter).where(Fighter.name == name)).scalars().all())
    if not rows:
        return None
    canonical = prefer_canonical(
        rows,
        source_key=lambda f: f.source,
        tiebreak_key=lambda f: f.id,
    )
    return canonical.id


# ── Populator helpers ───────────────────────────────────────────────────────


def _populate_elo(
    session: Session,
    fa_id: int,
    fb_id: int,
    feats: dict[str, float],
) -> tuple[float, float]:
    """Section 1: Elo differentials (3 features) + cross-domain (2 features).

    Returns ``(elo_a_overall, elo_b_overall)`` for the odds_elo_divergence
    derivation later in ``_populate_odds``.
    """
    elo_a_overall = _get_latest_elo(session, fa_id, "overall")
    elo_b_overall = _get_latest_elo(session, fb_id, "overall")
    elo_a_striking = _get_latest_elo(session, fa_id, "striking")
    elo_b_striking = _get_latest_elo(session, fb_id, "striking")
    elo_a_grappling = _get_latest_elo(session, fa_id, "grappling")
    elo_b_grappling = _get_latest_elo(session, fb_id, "grappling")

    feats["elo_overall_diff"] = elo_a_overall - elo_b_overall
    feats["elo_striking_diff"] = elo_a_striking - elo_b_striking
    feats["elo_grappling_diff"] = elo_a_grappling - elo_b_grappling
    feats["a_striking_vs_b_grappling"] = elo_a_striking - elo_b_grappling
    feats["a_grappling_vs_b_striking"] = elo_a_grappling - elo_b_striking

    return elo_a_overall, elo_b_overall


def _populate_performance(
    session: Session,
    fa_id: int,
    fb_id: int,
    feats: dict[str, float],
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """Section 2: Performance feature differentials (20 features, ``_diff`` suffix).

    Lifts the differential math from ``predictor.py:_build_feature_vector``
    (lines 293-301) — same NaN-on-either-side semantics as
    ``feature_matrix.py:414-420``.

    Returns the per-fighter latest features dicts so the caller can reuse
    them for ``_populate_network`` without re-querying the DB.
    """
    feats_a = _get_latest_computed_features(session, fa_id)
    feats_b = _get_latest_computed_features(session, fb_id)
    for feat_key in PERFORMANCE_FEATURE_KEYS:
        val_a = feats_a.get(feat_key)
        val_b = feats_b.get(feat_key)
        if val_a is not None and val_b is not None:
            feats[f"{feat_key}_diff"] = val_a - val_b
        # else: stays NaN per Pattern D
    return feats_a, feats_b


def _populate_network(
    feats: dict[str, float],
    feats_a_latest: dict[str, float | None],
    feats_b_latest: dict[str, float | None],
) -> None:
    """Section 5: Opponent-network differentials (3 features, NET-01/02).

    Phase 16-03 — operator-approved pan-mma + MOV-weighted config per the
    NET-00 gsd-checkpoint. Mirrors ``feature_matrix.py`` Section 14 exactly
    (Pitfall #12 train/predict parity); both call into
    ``features.network.compute_network_diff_features`` for the math.

    ``feats_a_latest`` / ``feats_b_latest`` are the latest
    ``ComputedFeature.features`` JSONB dicts as returned by
    ``_get_latest_computed_features`` (which lifts ``pagerank`` /
    ``sos_2hop`` / ``is_debutant_in_graph`` keys when present).

    Per D-06: when the per-fighter feature is missing/None (debutant or
    pre-Phase-16 snapshot), value flows through as NaN — Pattern D
    (NEVER 0.0 for missing).
    """
    from ufc_prediction.features.network import compute_network_diff_features

    nan = float("nan")

    def _net_input(latest: dict[str, float | None]) -> dict[str, float]:
        """Coerce per-fighter latest features into the NET-* shape, NaN on miss."""
        out: dict[str, float] = {}
        for k in ("pagerank", "sos_2hop", "is_debutant_in_graph"):
            v = latest.get(k) if latest else None
            out[k] = float(v) if v is not None else nan
        return out

    diffs = compute_network_diff_features(
        _net_input(feats_a_latest),
        _net_input(feats_b_latest),
    )
    feats["pagerank_diff"] = diffs["pagerank_diff"]
    feats["sos_2hop_diff"] = diffs["sos_2hop_diff"]
    feats["is_debutant_in_graph_diff"] = diffs["is_debutant_in_graph_diff"]


# ── Phase 23 v2.2 REF live-path helpers ─────────────────────────────────────
#
# Per CONTEXT D-10 + Pitfall #12: this module imports compute_ref_rates_shrunk
# from features_v22.ref — the SAME helper the training path (feature_matrix.py)
# uses. The DB-shaped inputs are built here in `_query_ref_state`; the pure
# compute lives in features_v22/ref.py (single source of truth).
#
# Strict pre-fight discipline (Pitfall #4): both ref_history and the global
# rate aggregate are restricted to ``event_date < the_event_date`` so the
# live path mirrors the training path's as-of-date guard.


def _query_ref_state(
    session: Session,
    referee_id: int | None,
    the_event_date: date,
) -> tuple[dict[int, list[dict]], dict[str, float]]:
    """Build the (ref_history, ref_global_rates) inputs for the REF compute.

    Returns:
        ref_history: ``{referee_id: [{"event_date": d, "method": m}, ...]}``
            filtered to ``event_date < the_event_date`` for the given
            ``referee_id``. Empty dict when ``referee_id is None``.
        ref_global_rates: ``{"finish": x, "decision": y, "no_action": z}``
            aggregated across ALL events with ``event_date < the_event_date``
            (live path enforces the same pre-fight cutoff as training for
            LIVE-03 parity per D-10).

    On DB error or unreachable session, returns empty history + zero-rate
    globals — caller (compute_ref_rates_shrunk) treats this as the Bayesian
    fallback and emits global_rates as-is.
    """
    nan_globals = {"finish": 0.0, "decision": 0.0, "no_action": 0.0}

    try:
        # Global rates: all fights at events strictly before the_event_date.
        global_stmt = (
            select(Fight.method, Event.date)
            .join(Event, Fight.event_id == Event.id)
            .where(Event.date < the_event_date)
        )
        counts = {"finish": 0, "decision": 0, "no_action": 0}
        total = 0
        for method, _event_date in session.execute(global_stmt).all():
            cat = classify_outcome(method)
            counts[cat] += 1
            total += 1

        global_rates = nan_globals if total == 0 else {k: v / total for k, v in counts.items()}

        if referee_id is None:
            return {}, global_rates

        # Per-referee history: fights at events officiated by this referee
        # strictly before the_event_date.
        ref_stmt = (
            select(Event.date, Fight.method)
            .join(Event, Fight.event_id == Event.id)
            .where(Event.referee_id == referee_id)
            .where(Event.date < the_event_date)
        )
        history: list[dict] = []
        for event_date_val, method in session.execute(ref_stmt).all():
            history.append({"event_date": event_date_val, "method": method})

        return ({referee_id: history}, global_rates)
    except Exception as exc:
        logger.warning("inference_features _query_ref_state failed: %s", exc)
        return {}, nan_globals


def _populate_ref(
    session: Session,
    feats: dict[str, float],
    referee_id: int | None,
    event_date_val: date,
) -> None:
    """Populate the 3 REF cols using the SAME helper feature_matrix.py uses.

    Pitfall #12 train/predict parity: this helper calls into
    ``features_v22.ref.compute_ref_rates_shrunk`` so the math is byte-identical
    to ``feature_matrix.py``. The DB-shaped inputs are built in
    ``_query_ref_state``; this module just wires them.
    """
    ref_history, ref_global_rates = _query_ref_state(
        session,
        referee_id,
        event_date_val,
    )
    rates = compute_ref_rates_shrunk(
        referee_id,
        event_date_val,
        ref_history,
        ref_global_rates,
    )
    feats["ref_finish_rate_shrunk"] = rates["ref_finish_rate_shrunk"]
    feats["ref_decision_rate_shrunk"] = rates["ref_decision_rate_shrunk"]
    feats["ref_no_action_rate_shrunk"] = rates["ref_no_action_rate_shrunk"]


# ── Phase 23 v2.2 TRAVEL live-path helpers (Plan 23-02) ──────────────────────
#
# Per CONTEXT D-10 + Pitfall #12: this module imports compute_travel_features
# from features_v22.travel — the SAME helper the training path
# (feature_matrix.py) uses. The DB-shaped venue lookups are built here via
# SQLAlchemy; the pure compute lives in features_v22/travel.py (single source
# of truth).
#
# Sentinel discipline (CONTEXT D-04 + Pitfall #3):
#   - prior_venue is None (debut fighter)        → 0  (sentinel)
#   - current event has no venue_id              → NaN (graceful degradation)
# These are DIFFERENT cases and must not be conflated.


def _query_current_venue(
    session: Session,
    event_id: int,
) -> dict | None:
    """Look up ``{lat, lon, timezone_iana}`` for the current event's venue.

    Returns ``None`` when the event has no ``venue_id`` set OR the Venue row
    is missing. ``None`` triggers NaN-padding for the 6 TRAVEL cols in the
    caller (graceful degradation per Pattern D).
    """
    try:
        stmt = (
            select(Venue.lat, Venue.lon, Venue.timezone_iana)
            .join(Event, Event.venue_id == Venue.id)
            .where(Event.id == event_id)
        )
        row = session.execute(stmt).first()
        if row is None:
            return None
        lat, lon, tz = row
        if lat is None or lon is None or tz is None:
            return None
        return {"lat": lat, "lon": lon, "timezone_iana": tz}
    except Exception as exc:
        logger.warning(
            "inference_features _query_current_venue failed: %s",
            exc,
        )
        return None


def _query_fighter_prior_venue(
    session: Session,
    fighter_id: int,
    before_date: date,
) -> dict | None:
    """Most recent prior fight's venue for ``fighter_id``, strictly before
    ``before_date``.

    Returns ``{"lat", "lon", "timezone_iana", "event_date"}`` or ``None``
    (debut fighter — no prior UFC fight). Strict ``<`` cutoff matches the
    training path's Pitfall #4 leakage guard.
    """
    try:
        stmt = (
            select(Venue.lat, Venue.lon, Venue.timezone_iana, Event.date)
            .join(Event, Event.venue_id == Venue.id)
            .join(Fight, Fight.event_id == Event.id)
            .where((Fight.fighter_a_id == fighter_id) | (Fight.fighter_b_id == fighter_id))
            .where(Event.date < before_date)
            .order_by(Event.date.desc())
            .limit(1)
        )
        row = session.execute(stmt).first()
        if row is None:
            return None
        lat, lon, tz, event_date_val = row
        if lat is None or lon is None or tz is None or event_date_val is None:
            return None
        return {
            "lat": lat,
            "lon": lon,
            "timezone_iana": tz,
            "event_date": event_date_val,
        }
    except Exception as exc:
        logger.warning(
            "inference_features _query_fighter_prior_venue failed: %s",
            exc,
        )
        return None


def _populate_travel(
    session: Session,
    feats: dict[str, float],
    fighter_a_id: int,
    fighter_b_id: int,
    event_id: int | None,
    event_date_val: date,
) -> None:
    """Populate the 6 TRAVEL cols using the SAME helper feature_matrix.py uses.

    Pitfall #12 train/predict parity: this helper calls into
    ``features_v22.travel.compute_travel_features`` so the math is byte-
    identical to ``feature_matrix.py``. The DB-shaped venue lookups are
    built here via SQLAlchemy queries; the pure compute lives in
    ``features_v22/travel.py`` (single source of truth).

    Sentinel discipline:
        - ``event_id is None`` or current event has no venue → 6 NaN
          (graceful degradation per Pattern D; preserves the dict-init
          NaN values for these 6 keys).
        - Fighter has no prior UFC fight (debut) → that fighter's
          ``travel_distance`` + ``tz_shift`` = 0 (D-04 sentinel).
    """
    if event_id is None:
        return  # NaN-init preserved
    curr_venue = _query_current_venue(session, event_id)
    if curr_venue is None:
        return  # NaN-init preserved (no usable current venue)
    prior_a = _query_fighter_prior_venue(session, fighter_a_id, event_date_val)
    prior_b = _query_fighter_prior_venue(session, fighter_b_id, event_date_val)
    travel_red = compute_travel_features(prior_a, curr_venue, event_date_val)
    travel_blue = compute_travel_features(prior_b, curr_venue, event_date_val)
    feats["travel_distance_miles_red"] = travel_red["travel_distance_miles"]
    feats["travel_distance_miles_blue"] = travel_blue["travel_distance_miles"]
    feats["travel_distance_miles_diff"] = (
        travel_red["travel_distance_miles"] - travel_blue["travel_distance_miles"]
    )
    feats["tz_shift_red_signed"] = travel_red["tz_shift_signed"]
    feats["tz_shift_blue_signed"] = travel_blue["tz_shift_signed"]
    feats["tz_shift_diff_signed"] = travel_red["tz_shift_signed"] - travel_blue["tz_shift_signed"]


# ── Phase 23 v2.2 META live-path helpers (Plan 23-03) ──────────────────────
#
# Per CONTEXT D-10 + Pitfall #12: this module imports META helpers from
# features_v22.meta — the SAME helpers feature_matrix.py uses. The DB-shaped
# inputs are built here via SQLAlchemy queries; the pure compute lives in
# features_v22/meta.py (single source of truth).
#
# Strict pre-fight discipline (Pitfall #4): all queries enforce
# ``event_date < the_event_date`` so the live path mirrors the training path's
# as-of-date guard.


def _query_fighter_prior_fight_date(
    session: Session,
    fighter_id: int,
    before_date: date,
) -> date | None:
    """Most recent prior fight event_date for ``fighter_id`` (strict <
    ``before_date``). ``None`` for debut fighters → ``layoff_days`` returns
    0 sentinel (Q4 + D-06)."""
    try:
        stmt = (
            select(Event.date)
            .join(Fight, Fight.event_id == Event.id)
            .where((Fight.fighter_a_id == fighter_id) | (Fight.fighter_b_id == fighter_id))
            .where(Event.date < before_date)
            .order_by(Event.date.desc())
            .limit(1)
        )
        result = session.scalar(stmt)
        return result
    except Exception as exc:
        logger.warning(
            "inference_features _query_fighter_prior_fight_date failed: %s",
            exc,
        )
        return None


def _query_elo_history(
    session: Session,
    fighter_id: int,
    before_date: date,
    limit: int = 6,
) -> list[dict]:
    """Return up to ``limit`` most-recent prior Elo snapshots for fighter.

    Each entry: ``{"elo_overall": x, "elo_striking": y, "elo_grappling": z}``.
    Strict ``Event.date < before_date`` cutoff (Pitfall #4). Returns
    chronological list (oldest first) so caller can call
    ``elo_velocity(history, window=5)`` directly.
    """
    try:
        # EloSnapshot has 3 rows per fight (overall/striking/grappling).
        # Pull the last `limit` fight_ids for this fighter, then collect
        # the 3 elo_type rows per fight.
        from sqlalchemy import distinct

        # Step 1: find the last `limit` distinct fight_dates for this fighter.
        date_stmt = (
            select(distinct(EloSnapshot.fight_date))
            .where(EloSnapshot.fighter_id == fighter_id)
            .where(EloSnapshot.fight_date < before_date)
            .order_by(EloSnapshot.fight_date.desc())
            .limit(limit)
        )
        fight_dates = [d for d in session.execute(date_stmt).scalars().all()]
        if not fight_dates:
            return []

        # Step 2: pull all elo_type snapshots for these dates.
        snap_stmt = (
            select(
                EloSnapshot.fight_date,
                EloSnapshot.elo_type,
                EloSnapshot.elo_after_shrinkage,
            )
            .where(EloSnapshot.fighter_id == fighter_id)
            .where(EloSnapshot.fight_date.in_(fight_dates))
        )
        # Group by fight_date.
        by_date: dict[date, dict[str, float]] = {}
        for fd, elo_type, elo_val in session.execute(snap_stmt).all():
            by_date.setdefault(fd, {})[elo_type] = float(elo_val)

        # Step 3: build chronological list (oldest first).
        sorted_dates = sorted(by_date.keys())
        result: list[dict] = []
        for fd in sorted_dates:
            entry = by_date[fd]
            result.append(
                {
                    "elo_overall": entry.get("overall", 1500.0),
                    "elo_striking": entry.get("striking", 1500.0),
                    "elo_grappling": entry.get("grappling", 1500.0),
                }
            )
        return result
    except Exception as exc:
        logger.warning(
            "inference_features _query_elo_history failed: %s",
            exc,
        )
        return []


def _query_division_state(
    session: Session,
    weight_class: str | None,
    before_date: date,
) -> tuple[dict[str, list[dict]], float]:
    """Return ``(div_history, global_finish_rate)`` for division finish-rate
    EB compute.

    The div_history (per-class fight list) carries chronological entries
    for the queried weight class — the helper
    ``division_finish_rate_shrunk`` applies the strict
    ``event_date < as_of_date`` cutoff internally (Pitfall #4 leakage
    guard at the call site).

    The ``global_finish_rate`` is aggregated across the FULL corpus (no
    temporal filter) to match the training-path discipline in
    ``feature_matrix._build_division_history`` (which also returns a
    corpus-wide single global). Marginal leakage from the global prior is
    accepted; per-class counts carry the strict pre-fight filter.
    """
    if weight_class is None:
        return {}, 0.0
    try:
        # Per-class history: chronological list for the queried weight class.
        # Global rate: corpus-wide aggregate (matches training-path).
        stmt = select(Fight.weight_class, Fight.method, Event.date).join(
            Event, Fight.event_id == Event.id
        )
        div_hist: dict[str, list[dict]] = {}
        total_finishes = 0
        total = 0
        for wc, method, evt_date in session.execute(stmt).all():
            if wc is not None:
                div_hist.setdefault(wc, []).append(
                    {
                        "event_date": evt_date,
                        "method": method,
                    }
                )
            total += 1
            if classify_outcome(method) == "finish":
                total_finishes += 1
        g = total_finishes / total if total > 0 else 0.0
        return div_hist, g
    except Exception as exc:
        logger.warning(
            "inference_features _query_division_state failed: %s",
            exc,
        )
        return {}, 0.0


def _query_division_mean_reach(
    session: Session,
    weight_class: str | None,
) -> float | None:
    """Mean reach_inches across fighters who have fought in ``weight_class``.

    Returns ``None`` when class is unknown or no fighters with known reach;
    caller treats as 0.0 → ``reach_diff_normalized`` → NaN (Pattern D).
    """
    if weight_class is None:
        return None
    try:
        from sqlalchemy import func as sa_func

        stmt = (
            select(sa_func.avg(Fighter.reach_inches))
            .select_from(Fighter)
            .join(
                Fight,
                (Fight.fighter_a_id == Fighter.id) | (Fight.fighter_b_id == Fighter.id),
            )
            .where(Fight.weight_class == weight_class)
            .where(Fighter.reach_inches.is_not(None))
        )
        result = session.scalar(stmt)
        return float(result) if result is not None else None
    except Exception as exc:
        logger.warning(
            "inference_features _query_division_mean_reach failed: %s",
            exc,
        )
        return None


def _query_fighter_division(
    session: Session,
    fighter_id: int,
) -> str | None:
    """Resolve a fighter's most recent weight_class (best-effort).

    For unknown / unsigned fighters returns ``None`` → division-prior
    falls back to global finish rate (Bayesian fallback).
    """
    try:
        stmt = (
            select(Fight.weight_class)
            .join(Event, Fight.event_id == Event.id)
            .where((Fight.fighter_a_id == fighter_id) | (Fight.fighter_b_id == fighter_id))
            .order_by(Event.date.desc())
            .limit(1)
        )
        return session.scalar(stmt)
    except Exception as exc:
        logger.warning(
            "inference_features _query_fighter_division failed: %s",
            exc,
        )
        return None


def _populate_meta(
    session: Session,
    feats: dict[str, float],
    fighter_a,
    fighter_b,
    event_date_val: date,
) -> None:
    """Populate the 9 META cols using SAME helpers feature_matrix.py uses.

    Pitfall #12 train/predict parity: all compute via features_v22.meta.
    Pitfall #5 fix: age uses event_date (already corrected in
    ``_populate_physical``; per-fighter ``age_at_fight_*`` here also uses
    event_date).

    Only per-fighter ``layoff_days_red``/``layoff_days_blue`` are written here.

    KNOWN GAP (review #12, corrects a stale note): the differential
    ``days_since_last_fight_diff`` is NOT populated at inference — it is not in
    ``_populate_performance``'s key set and is never assigned, so it stays NaN.
    The base xgb_v2 model tolerates this via XGBoost's NaN default branch, but it
    blocks the META-V22 stacker (whose StandardScaler cannot ingest NaN), so the
    meta currently skips to the base prob on real matchups. Closing this gap
    changes base-model inference inputs and must be gate-validated (follow-up).
    """
    # Layoff (per-fighter only): query each fighter's prior fight date.
    prior_a = _query_fighter_prior_fight_date(
        session,
        fighter_a.id,
        event_date_val,
    )
    prior_b = _query_fighter_prior_fight_date(
        session,
        fighter_b.id,
        event_date_val,
    )
    feats["layoff_days_red"] = layoff_days(event_date_val, prior_a)
    feats["layoff_days_blue"] = layoff_days(event_date_val, prior_b)
    # NB: layoff_days_diff intentionally NOT written (Q4 + D-07).

    # Age (Pitfall #5 — uses event_date).
    feats["age_at_fight_red"] = age_at_fight(
        getattr(fighter_a, "date_of_birth", None),
        event_date_val,
    )
    feats["age_at_fight_blue"] = age_at_fight(
        getattr(fighter_b, "date_of_birth", None),
        event_date_val,
    )

    # Elo velocity: query last N+1 snapshots (window=5 → need 6 entries).
    eh_a = _query_elo_history(session, fighter_a.id, event_date_val, limit=6)
    eh_b = _query_elo_history(session, fighter_b.id, event_date_val, limit=6)

    def _vel_diff(key: str) -> float:
        va = elo_velocity([s[key] for s in eh_a], window=5)
        vb = elo_velocity([s[key] for s in eh_b], window=5)
        if va != va or vb != vb:  # NaN propagation
            return float("nan")
        return va - vb

    feats["elo_overall_velocity_diff"] = _vel_diff("elo_overall")
    feats["elo_striking_velocity_diff"] = _vel_diff("elo_striking")
    feats["elo_grappling_velocity_diff"] = _vel_diff("elo_grappling")

    # Division finish rate: resolve fighter's most recent division.
    weight_class = _query_fighter_division(session, fighter_a.id) or _query_fighter_division(
        session, fighter_b.id
    )
    div_hist, global_finish_rate = _query_division_state(
        session,
        weight_class,
        event_date_val,
    )
    feats["division_finish_rate_shrunk"] = division_finish_rate_shrunk(
        weight_class,
        event_date_val,
        div_hist,
        global_finish_rate,
        k_shrink=50.0,
    )

    # Reach normalized.
    mean_reach = _query_division_mean_reach(session, weight_class) or 0.0
    feats["reach_diff_normalized"] = reach_diff_normalized(
        getattr(fighter_a, "reach_inches", None),
        getattr(fighter_b, "reach_inches", None),
        mean_reach,
    )


def _populate_physical(
    fighter_a,
    fighter_b,
    feats: dict[str, float],
    event_date: date,
) -> None:
    """Section 3: Physical differentials (4 features) + stance (1 feature).

    Reads attributes directly off the Fighter ORM rows (already loaded by
    the predictor's ``_resolve_fighter`` call). Same nullable semantics as
    ``feature_matrix.py:422-458`` — NaN on either side missing.

    Phase 23 Pitfall #5 fix: ``age_diff`` is computed from ``event_date``
    (NOT ``date.today()``) so backtests against historical events compute
    the correct age. The age at a historical fight is determined by the
    event_date, not by the current calendar date. Prior to this fix, the
    inference path diverged from the training path
    (``feature_matrix.py:650-659`` already uses ``event_date``) for any
    ``event_date != today`` — a silent parity break for OOF generation.
    """
    for attr, key in (
        ("height_inches", "height_diff"),
        ("reach_inches", "reach_diff"),
        ("leg_reach_inches", "leg_reach_diff"),
    ):
        val_a = getattr(fighter_a, attr, None)
        val_b = getattr(fighter_b, attr, None)
        if val_a is not None and val_b is not None:
            feats[key] = val_a - val_b

    dob_a = getattr(fighter_a, "date_of_birth", None)
    dob_b = getattr(fighter_b, "date_of_birth", None)
    if dob_a is not None and dob_b is not None:
        # Phase 23 Pitfall #5 fix: event_date NOT date.today().
        age_a = (event_date - dob_a).days / 365.25
        age_b = (event_date - dob_b).days / 365.25
        feats["age_diff"] = age_a - age_b

    feats["stance_matchup"] = encode_stance_matchup(
        getattr(fighter_a, "stance", None),
        getattr(fighter_b, "stance", None),
    )


def _populate_odds(
    feats: dict[str, float],
    live_odds: MatchupOdds | None,
    cached_a: dict | None,
    cached_b: dict | None,
    elo_a_overall: float,
    elo_b_overall: float,
) -> None:
    """Section 4: 5-feature Phase 15.1 odds block.

    Mirrors ``feature_matrix.py:560-625`` exactly (Gotcha 2). Per D-09,
    if ``live_odds`` is provided we derive implied probabilities from the
    raw American moneylines; otherwise we use the cached implied probs.
    Per Pattern D / Pitfall #3: missing data → NaN (NEVER 0.0).
    """
    op_a: float | None = None
    op_b: float | None = None
    cl_a: float | None = None
    cl_b: float | None = None

    if live_odds is not None:
        # Live BFO path: A's row contains both A and B moneylines after the
        # ingest pipeline. The bfo_live single-shot only populates A's
        # moneylines from the parsed fighter-page row; B-side often arrives
        # as None and must NaN out per Pattern D.
        #
        # Bad-data resilience (review #12): the devig helpers raise
        # InvalidMoneylineError on an out-of-domain moneyline (|ml| < 100). Per
        # Pattern D ("missing data → NaN, never crash"), a corrupt odds value must
        # leave the odds feats NaN, not raise out of predict(). Catch, log
        # (mirroring the ingest path), and degrade.
        if live_odds.fighter_a_opening is not None and live_odds.fighter_b_opening is not None:
            try:
                op_a, op_b = devig_proportional(
                    live_odds.fighter_a_opening,
                    live_odds.fighter_b_opening,
                )
            except InvalidMoneylineError as exc:
                logger.warning("live opening odds NaN-degraded (bad moneyline): %s", exc)
                op_a = op_b = None
        if (
            live_odds.fighter_a_closing_min is not None
            and live_odds.fighter_a_closing_max is not None
            and live_odds.fighter_b_closing_min is not None
            and live_odds.fighter_b_closing_max is not None
        ):
            try:
                cl_a, cl_b = devig_closing_range(
                    live_odds.fighter_a_closing_min,
                    live_odds.fighter_a_closing_max,
                    live_odds.fighter_b_closing_min,
                    live_odds.fighter_b_closing_max,
                )
            except InvalidMoneylineError as exc:
                logger.warning("live closing odds NaN-degraded (bad moneyline): %s", exc)
                cl_a = cl_b = None
    elif cached_a is not None and cached_b is not None:
        op_a = cached_a.get("opening_implied_prob")
        op_b = cached_b.get("opening_implied_prob")
        cl_a = cached_a.get("closing_implied_prob")
        cl_b = cached_b.get("closing_implied_prob")

    # opening_prob_diff
    if op_a is not None and op_b is not None:
        feats["opening_prob_diff"] = op_a - op_b
    # closing_prob_diff
    if cl_a is not None and cl_b is not None:
        feats["closing_prob_diff"] = cl_a - cl_b
    # line_movement_diff + sharp_money_signal — both require all 4 vals
    if op_a is not None and op_b is not None and cl_a is not None and cl_b is not None:
        line_move_diff = (cl_a - op_a) - (cl_b - op_b)
        feats["line_movement_diff"] = line_move_diff
        feats["sharp_money_signal"] = abs(line_move_diff)
    # odds_elo_divergence: needs cl_a (market for A) AND elo overall both sides
    if cl_a is not None:
        elo_diff = elo_a_overall - elo_b_overall
        elo_prob_a = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        feats["odds_elo_divergence"] = cl_a - elo_prob_a


# ── Public entry point ──────────────────────────────────────────────────────


def build(
    session: Session,
    fighter_a,
    fighter_b,
    event_date: date,
    *,
    live_odds: MatchupOdds | None = None,
    include_net: bool | None = None,
    feature_set: str = "v1.0",
    referee_id: int | None = None,
    event_id: int | None = None,
) -> np.ndarray:
    """Build a ``(1, len(cols))`` feature vector from latest snapshots
    + injected live/cached odds.

    Per Phase 23 D-09: ``feature_set`` is the new public knob.
        - ``"v1.0"``        → 75 cols (includes 3 NET-* tail).
        - ``"v2.1-no-net"`` → 72 cols (xgb_v2 baseline, no NET).
        - ``"v2.2"``        → 90 cols (72 + 3 REF + 6 TRAVEL + 9 META;
                             TRAVEL+META NaN until Plans 23-02/03 wire them).

    ``include_net`` is the Phase 18 back-compat shim:
        - ``include_net=True``  → ``feature_set="v1.0"``
        - ``include_net=False`` → ``feature_set="v2.1-no-net"``
        - ``include_net=None``  → respect ``feature_set`` as-is (default).

    Args:
        session: Open SQLAlchemy session for snapshot lookups.
        fighter_a, fighter_b: Resolved Fighter ORM rows (post-dedup).
        event_date: As-of date for cache lookup. Snapshot reads take the
            most recent snapshot regardless of ``event_date`` since they
            are immutable per fight.
        live_odds: Output of ``bfo_live.fetch_matchup_odds``. ``None`` falls
            back to cache; cache miss falls back to NaN per Pattern D.
        feature_set: Which column list to materialize (v1.0/v2.1-no-net/v2.2).
        referee_id: Phase 23 REF input. Only used when ``feature_set='v2.2'``;
            ``None`` triggers the Bayesian fallback (all 3 REF cols collapse
            to ``global_rates``).
        event_id: Phase 23 TRAVEL input (Plan 23-02). Only used when
            ``feature_set='v2.2'``; ``None`` or unresolvable venue triggers
            graceful NaN degradation for the 6 TRAVEL cols. Debut fighters
            (no prior UFC fight) get the 0 sentinel per CONTEXT D-04.

    Returns:
        ``np.ndarray`` of shape ``(1, len(cols))``, dtype ``float64``.
        Columns positioned per the resolved column list (Pitfall #12 lock).

    Raises:
        RuntimeError: If a populator emits a feature key not in the resolved
            ``cols`` list (positional drift safeguard, lifted from
            ``predictor.py:344-352``).
    """
    # Phase 18 back-compat shim.
    if include_net is not None:
        feature_set = "v1.0" if include_net else "v2.1-no-net"
    # Internal flag for the NET-block gate below.
    _include_net = feature_set == "v1.0"

    cols = get_feature_columns(feature_set=feature_set)
    feats: dict[str, float] = {col: float("nan") for col in cols}

    # Section 1 & 2: Elo + performance differentials
    elo_a_overall, elo_b_overall = _populate_elo(
        session,
        fighter_a.id,
        fighter_b.id,
        feats,
    )
    feats_a_latest, feats_b_latest = _populate_performance(
        session,
        fighter_a.id,
        fighter_b.id,
        feats,
    )

    # Section 3: Physical + stance (Pitfall #5 fix — age uses event_date)
    _populate_physical(fighter_a, fighter_b, feats, event_date)

    # Section 4: 5-feature odds block (Gotcha 2). Live takes precedence;
    # cache is consulted only when live_odds is None.
    cached_a, cached_b = (None, None)
    if live_odds is None:
        cached_a, cached_b = _get_cached_odds(
            session,
            fighter_a.id,
            fighter_b.id,
            event_date,
        )
    _populate_odds(
        feats,
        live_odds,
        cached_a,
        cached_b,
        elo_a_overall,
        elo_b_overall,
    )

    # Section 5: 3-feature opponent-network block (NET-01/02, Phase 16-03).
    # Reuses the same latest-feature dicts read by _populate_performance —
    # NET-* keys live in the same ComputedFeature.features JSONB written by
    # FeatureComputer Pass 4. Train/predict parity (Pitfall #12) is enforced
    # by sharing compute_network_diff_features with feature_matrix.py.
    # Phase 18 NET-V2-01: when feature_set != "v1.0", this block is SKIPPED.
    if _include_net:
        _populate_network(feats, feats_a_latest, feats_b_latest)

    # Section 6: Phase 23 v2.2 REF + TRAVEL + META cols.
    # Per CONTEXT D-10: inference_features.py imports compute helpers
    # from features_v22.{ref,travel,meta} — the SAME helpers
    # feature_matrix.py uses (Pitfall #12 LIVE-03 parity).
    if feature_set == "v2.2":
        _populate_ref(session, feats, referee_id, event_date)
        # Plan 23-02 TRAVEL: 6 cols at FEATURE_COLUMNS_V22 indices 75-80.
        # event_id is None → preserves NaN-init for the 6 cols (graceful
        # degradation per Pattern D); debut fighter → 0 sentinel per D-04.
        _populate_travel(
            session,
            feats,
            fighter_a.id,
            fighter_b.id,
            event_id,
            event_date,
        )
        # Plan 23-03 META: 9 cols at FEATURE_COLUMNS_V22 indices 81-89.
        # Pitfall #5 fixed: age_at_fight_* uses event_date.
        # Q4 RESOLVED: no layoff_days_diff (days_since_last_fight_diff at
        # FEATURE_COLUMNS_NO_NET[61] is the canonical differential).
        _populate_meta(session, feats, fighter_a, fighter_b, event_date)

    # Section 7: strict column-order materialization (Pitfall #12 guard,
    # lifted from predictor.py:332-340).
    unknown_keys = set(feats) - set(cols)
    if unknown_keys:
        raise RuntimeError(
            "inference_features populated keys not in model "
            f"feature_columns: {sorted(unknown_keys)}"
        )
    row: list[Any] = [feats[col] for col in cols]
    return np.array([row], dtype=np.float64)


# ── Module-level Phase 23 D-10 load-time guard ──────────────────────────────
#
# Mirror of the assert in feature_matrix.py. When feature_set='v2.2' is used,
# this module emits a row of exactly FEATURE_COLUMNS_V22 cols in exactly that
# order. The dynamic per-fight guard above (unknown_keys + ordered row) is the
# runtime check; the static guard below verifies the constant's shape at
# import time so a config drift fails fast.
_EXPECTED_V22_NCOLS = len(FEATURE_COLUMNS_V22)
assert _EXPECTED_V22_NCOLS == 72 + 3 + 6 + 9, (
    f"FEATURE_COLUMNS_V22 length drift: expected 90 (72+3+6+9), "
    f"got {_EXPECTED_V22_NCOLS}. Anti-Pattern #8 violation."
)
