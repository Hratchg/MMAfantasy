"""Predict-time twin of feature_matrix.py (LIVE-02, closes 44/72 NaN-pad gap).

Replays both fighters' stored history as of ``event_date`` (performance
accumulators, Elo, career stats) and builds a ``(1, len(FEATURE_COLUMNS))``-shaped
feature vector for a single matchup.

Per CONTEXT.md D-12: this module REPLACES the NaN-pad block at
``predictor.py:289-352``. The NaN-pad block is deleted in 16-02 Task 4.

Per Gotcha 2 / Pattern Map: the 5-feature Phase 15.1 odds block at
``feature_matrix.py:560-625`` is replicated here column-for-column.

Per Pattern D: NaN — never 0.0 — for missing odds. XGBoost handles native
via sparsity-aware split finding (D-04(P14)).

Per Pitfall #12: column ordering is locked by ``FEATURE_COLUMNS``; a strict
``unknown_keys`` guard prevents silent positional drift.

The DB-reading helpers (``_get_latest_elo``, ``_get_pre_fight_performance``,
``_get_cached_odds``, ``_load_career_inputs``) are module-level so tests can
monkey-patch them without spinning up Postgres.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np
import psycopg
from sqlalchemy import case, select
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session

from ufc_prediction.dedup.source_priority import prefer_canonical
from ufc_prediction.elo.asof import RatedFight, pre_fight_rating
from ufc_prediction.elo.config import EloConfig
from ufc_prediction.elo.engine import _NON_TRANSFER_DIVISIONS
from ufc_prediction.elo.seed_store import SEED_TABLE, load_seeds_from_db
from ufc_prediction.features import queries as feature_queries
from ufc_prediction.features.compute import FeatureComputer
from ufc_prediction.ml import queries as ml_queries
from ufc_prediction.ml.config import (
    FEATURE_COLUMNS_V22,
    PERFORMANCE_FEATURE_KEYS,
    MLConfig,
    encode_stance_matchup,
    get_feature_columns,
)
from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler, compute_division_medians
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


# Only a non-empty seed map is cached: an empty result (empty or not yet
# migrated debutant_seed_inputs table) is re-checked on every call so seeds
# backfilled while a long-running API process is up get picked up instead of
# being masked for the process life.
_debutant_seeds_cache: dict[int, float] | None = None
_warned_missing_seeds: bool = False


def _load_debutant_seeds(session: Session) -> dict[int, float]:
    """Debutant Elo seeds (DEBUT-V25-03), from the same table ``elo compute`` reads.

    The ``debutant_seed_inputs`` table is the single source of truth (D3
    option C), so the Docker serving image needs no Sherdog CSV. An empty
    table, or a missing one (``alembic upgrade head`` not applied), yields
    ``{}``: serving still works, but debutants get the flat-1500 default while
    the stored ``elo_before`` substrate was built with seeds (train/serve skew
    on ``elo_overall_diff``), so the miss is logged at ERROR (once per
    process). The query runs in a SAVEPOINT so a missing table cannot abort
    the request's transaction.
    """
    global _debutant_seeds_cache, _warned_missing_seeds
    if _debutant_seeds_cache is not None:
        return _debutant_seeds_cache
    problem = f"the {SEED_TABLE} table is empty"
    try:
        with session.begin_nested():
            seeds = load_seeds_from_db(session)
    except ProgrammingError as exc:
        if not isinstance(getattr(exc, "orig", None), psycopg.errors.UndefinedTable):
            raise
        seeds = {}
        problem = f"the {SEED_TABLE} table does not exist (run `alembic upgrade head`)"
    if seeds:
        _debutant_seeds_cache = seeds
    elif not _warned_missing_seeds:
        _warned_missing_seeds = True
        logger.error(
            "No debutant Elo seeds loaded (%s; load them with `ufc db "
            "backfill-pre-ufc-seeds`): debutant overall Elo falls back to flat 1500 "
            "while the stored elo_snapshots substrate is seeded, skewing "
            "elo_overall_diff for debutants.",
            problem,
        )
    return seeds


def _load_elo_history(
    session: Session,
    fighter_id: int,
    elo_type: str,
    before_date: date,
) -> list[RatedFight]:
    """All of a fighter's ``elo_type`` snapshots dated strictly before
    ``before_date``, oldest first — the input to ``pre_fight_rating``."""
    stmt = (
        select(EloSnapshot.fight_date, EloSnapshot.division, EloSnapshot.elo_after)
        .where(EloSnapshot.fighter_id == fighter_id)
        .where(EloSnapshot.elo_type == elo_type)
        .where(EloSnapshot.fight_date < before_date)
        .order_by(EloSnapshot.fight_date, EloSnapshot.fight_id)
    )
    return [
        RatedFight(fight_date=row[0], division=row[1], elo_after=float(row[2]))
        for row in session.execute(stmt).all()
    ]


def _get_latest_elo(
    session: Session,
    fighter_id: int,
    elo_type: str,
    as_of: date | None = None,
    division: str | None = None,
) -> float:
    """Return the rating training would have read as ``elo_before`` for a
    fight of ``fighter_id`` in ``division`` on ``as_of``.

    Training (``queries.load_elo_features``) reads ``EloSnapshot.elo_before``:
    the raw per-division rating after the engine's pre-fight inactivity
    regression and division transfer. This used to read the latest
    ``elo_after_shrinkage`` instead — a post-fight, Bayesian-shrunk display
    number with no regression or transfer — so ``elo_*_diff`` at serve time
    lived on a different scale than at training time. It now replays the
    fighter's stored history through ``elo.asof.pre_fight_rating`` with the
    engine's own semantics per Elo type:

    - ``overall``: regression + transfer, debutant seed on a fresh key.
    - ``striking`` / ``grappling``: regression + transfer, no seed
      (``DomainEloComputer`` keeps per-domain bookkeeping).

    ``as_of`` defaults to today; ``division`` ``None`` resolves to the
    fighter's most recent transferable division.

    The name is kept for the tests that monkeypatch this symbol.
    """
    as_of = as_of or date.today()
    history = _load_elo_history(session, fighter_id, elo_type, as_of)
    seed: float | None = (
        _load_debutant_seeds(session).get(fighter_id) if elo_type == "overall" else None
    )
    return pre_fight_rating(
        history,
        as_of,
        division,
        config=EloConfig(),
        seed=seed,
    )


_NET_KEYS: tuple[str, ...] = ("pagerank", "sos_2hop", "is_debutant_in_graph")

# ``Session.info`` key for the per-session performance substrate cache.
_PERFORMANCE_SUBSTRATE_KEY = "ufc_prediction.inference_features.performance_substrate"


@dataclass(frozen=True)
class PerformanceSubstrate:
    """What ``features compute`` reads, plus the league means it derives.

    ``fights`` / ``round_stats`` are the ``features.queries`` loader outputs
    (every source, as ``features compute`` loads them); ``league_means`` are
    the Pass 2 shrinkage targets over that whole corpus.
    """

    fights: list[dict[str, Any]]
    round_stats: dict[int, list[dict[str, Any]]]
    league_means: dict[str, float]


def _load_performance_substrate(session: Session) -> PerformanceSubstrate:
    """Load the ``features compute`` inputs once per session.

    The league means need a Pass 1 over the whole corpus (~1-2 s on the
    full DB), so the result is cached in ``session.info``: a card or an
    order-invariant request (several ``build`` calls) pays for it once, and
    a new session sees fresh data.
    """
    cached = session.info.get(_PERFORMANCE_SUBSTRATE_KEY)
    if isinstance(cached, PerformanceSubstrate):
        return cached
    fights = feature_queries.load_fights_with_duration(session)
    round_stats = feature_queries.load_all_round_stats(session)
    substrate = PerformanceSubstrate(
        fights=fights,
        round_stats=round_stats,
        league_means=FeatureComputer().league_means(fights, round_stats),
    )
    session.info[_PERFORMANCE_SUBSTRATE_KEY] = substrate
    return substrate


def _serve_keys(features: dict[str, Any] | None) -> dict[str, float | None]:
    """The 20 performance keys + 3 NET keys; empty for a debutant (no row)."""
    if not features:
        return {}
    return {key: features.get(key) for key in (*PERFORMANCE_FEATURE_KEYS, *_NET_KEYS)}


def _get_pre_fight_performance(
    session: Session,
    fa_id: int,
    fb_id: int,
    event_date: date,
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """Both fighters' performance + NET snapshot for a fight on ``event_date``.

    ``features compute`` stores each ``computed_features`` row as the
    PRE-fight snapshot of the fight it is keyed to, so a fighter's newest
    row predates their last fight and a one-fight fighter has none. Reading
    the newest row therefore served every performance column one fight
    stale (all-NaN for a sophomore) and, with no date cutoff, leaked later
    fights into a historical ``event_date``. Training reads the snapshot
    keyed to the target fight.

    This replays ``FeatureComputer`` over both fighters' fights dated
    strictly before ``event_date`` plus the upcoming fight
    (``FeatureComputer.compute_upcoming``), shrinking with the corpus-wide
    league means, so the result is the row ``features compute`` would store
    for this fight. Returns ``({}, …)`` for a debutant (NaN per Pattern D),
    exactly as training has no row for a debut.
    """
    substrate = _load_performance_substrate(session)
    prior = [
        f
        for f in substrate.fights
        if f["event_date"] < event_date
        and (
            fa_id in (f["fighter_a_id"], f["fighter_b_id"])
            or fb_id in (f["fighter_a_id"], f["fighter_b_id"])
        )
    ]
    snapshots = FeatureComputer().compute_upcoming(
        prior,
        substrate.round_stats,
        fa_id,
        fb_id,
        event_date,
        substrate.league_means,
        network_fights=substrate.fights,
    )
    return _serve_keys(snapshots.get(fa_id)), _serve_keys(snapshots.get(fb_id))


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


def _query_scheduled_bout(
    session: Session,
    fa_id: int,
    fb_id: int,
    event_date: date,
) -> tuple[str, int, bool] | None:
    """``(weight_class, num_rounds, is_title_fight)`` of the stored ``Fight``
    row for ``(A, B)`` on ``event_date``, either orientation.

    This is the bout context training reads off the same row. A ufcstats row
    wins over a kaggle duplicate. Returns ``None`` when no row exists (an
    unscheduled hypothetical matchup) or on any lookup failure.
    """
    try:
        stmt = (
            select(Fight.weight_class, Fight.num_rounds, Fight.is_title_fight)
            .join(Event, Fight.event_id == Event.id)
            .where(
                ((Fight.fighter_a_id == fa_id) & (Fight.fighter_b_id == fb_id))
                | ((Fight.fighter_a_id == fb_id) & (Fight.fighter_b_id == fa_id))
            )
            .where(Event.date == event_date)
            .order_by(case((Event.source == "ufcstats", 0), else_=1), Fight.id)
            .limit(1)
        )
        row = session.execute(stmt).first()
        if row is None:
            return None
        weight_class, num_rounds, is_title_fight = row
        if not isinstance(weight_class, str) or not weight_class:
            return None
        return (
            weight_class,
            num_rounds if isinstance(num_rounds, int) else 3,
            bool(is_title_fight) if isinstance(is_title_fight, bool) else False,
        )
    except Exception as exc:
        logger.warning("inference_features scheduled-bout lookup failed: %s", exc)
        return None


def _query_division_physical_medians(
    session: Session,
    weight_class: str | None,
    cutoff: date,
) -> dict[str, float]:
    """Training's D-01 imputation medians for one division.

    Feeds ``feature_matrix.compute_division_medians`` exactly what training
    does for ``weight_class``: the fighters of ufcstats fights with a winner
    in that division before ``cutoff``, with their physicals. Returns ``{}``
    when the division has no such fights (training then leaves the
    difference NaN) or on any lookup failure.
    """
    if not weight_class:
        return {}
    try:
        fight_stmt = _training_scope(
            select(Fight.fighter_a_id, Fight.fighter_b_id, Event.date)
            .join(Event, Fight.event_id == Event.id)
            .where(Fight.weight_class == weight_class)
            .where(Event.date < cutoff)
        )
        records = [
            {
                "fighter_a_id": a_id,
                "fighter_b_id": b_id,
                "event_date": event_date,
                "weight_class": weight_class,
            }
            for a_id, b_id, event_date in session.execute(fight_stmt).all()
        ]
        if not records:
            return {}
        fighter_ids = {r["fighter_a_id"] for r in records} | {r["fighter_b_id"] for r in records}
        phys_stmt = select(
            Fighter.id, Fighter.height_inches, Fighter.reach_inches, Fighter.leg_reach_inches
        ).where(Fighter.id.in_(fighter_ids))
        physicals = {
            fid: {"height_inches": h, "reach_inches": r, "leg_reach_inches": lr}
            for fid, h, r, lr in session.execute(phys_stmt).all()
        }
        return compute_division_medians(physicals, records, cutoff).get(weight_class, {})
    except Exception as exc:
        logger.warning("inference_features division-median lookup failed: %s", exc)
        return {}


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
    as_of: date | None = None,
    division: str | None = None,
) -> tuple[float, float]:
    """Section 1: Elo differentials (3 features) + cross-domain (2 features).

    Returns ``(elo_a_overall, elo_b_overall)`` for the odds_elo_divergence
    derivation later in ``_populate_odds``.
    """
    elo_a_overall = _get_latest_elo(session, fa_id, "overall", as_of, division)
    elo_b_overall = _get_latest_elo(session, fb_id, "overall", as_of, division)
    elo_a_striking = _get_latest_elo(session, fa_id, "striking", as_of, division)
    elo_b_striking = _get_latest_elo(session, fb_id, "striking", as_of, division)
    elo_a_grappling = _get_latest_elo(session, fa_id, "grappling", as_of, division)
    elo_b_grappling = _get_latest_elo(session, fb_id, "grappling", as_of, division)

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
    event_date: date,
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """Section 2: Performance feature differentials (20 features, ``_diff`` suffix).

    Lifts the differential math from ``predictor.py:_build_feature_vector``
    (lines 293-301) — same NaN-on-either-side semantics as
    ``feature_matrix.py:414-420``. The per-fighter snapshots are the
    pre-fight state as of ``event_date`` (``_get_pre_fight_performance``).

    Returns the per-fighter snapshot dicts so the caller can reuse them for
    ``_populate_network`` without replaying again.
    """
    feats_a, feats_b = _get_pre_fight_performance(session, fa_id, fb_id, event_date)
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

    ``feats_a_latest`` / ``feats_b_latest`` are the per-fighter pre-fight
    snapshots returned by ``_get_pre_fight_performance`` (which carry the
    ``pagerank`` / ``sos_2hop`` / ``is_debutant_in_graph`` keys).

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


def _training_scope(stmt: Any) -> Any:
    """Restrict a Fight⋈Event select to the fights training sees.

    ``ml.queries.load_fight_records`` keeps ufcstats fights with a winner
    (Plan 28-04 dedup). Without this the v2.2 serve helpers counted kaggle
    duplicates (~1.95x) and NC / scheduled rows that training never sees.
    """
    return stmt.where(Event.source == "ufcstats").where(Fight.winner_id.is_not(None))


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
        global_stmt = _training_scope(
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
        ref_stmt = _training_scope(
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
    training path's Pitfall #4 leakage guard. Like
    ``feature_matrix._build_fighter_prior_venues``, only training-scope
    fights at an event with a usable venue count.
    """
    try:
        stmt = _training_scope(
            select(Venue.lat, Venue.lon, Venue.timezone_iana, Event.date)
            .join(Event, Event.venue_id == Venue.id)
            .join(Fight, Fight.event_id == Event.id)
            .where((Fight.fighter_a_id == fighter_id) | (Fight.fighter_b_id == fighter_id))
            .where(Event.date < before_date)
            .where(Venue.lat.is_not(None))
            .where(Venue.lon.is_not(None))
            .where(Venue.timezone_iana.is_not(None))
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
        stmt = _training_scope(
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

    Reads ``elo_before`` — the same column training's
    ``feature_matrix._build_elo_histories`` sees via ``load_elo_features`` —
    not the shrunk post-fight display rating.
    """
    try:
        # EloSnapshot has 3 rows per fight (overall/striking/grappling).
        # Pull the last `limit` fight_ids for this fighter, then collect
        # the 3 elo_type rows per fight.
        from sqlalchemy import distinct

        # Step 1: find the last `limit` distinct fight_dates for this fighter.
        # Only snapshots of training-scope fights (ufcstats, winner present),
        # the same fights ``_build_elo_histories`` walks.
        date_stmt = _training_scope(
            select(distinct(EloSnapshot.fight_date))
            .join(Fight, EloSnapshot.fight_id == Fight.id)
            .join(Event, Fight.event_id == Event.id)
            .where(EloSnapshot.fighter_id == fighter_id)
            .where(EloSnapshot.fight_date < before_date)
            .order_by(EloSnapshot.fight_date.desc())
            .limit(limit)
        )
        fight_dates = [d for d in session.execute(date_stmt).scalars().all()]
        if not fight_dates:
            return []

        # Step 2: pull all elo_type snapshots for these dates.
        snap_stmt = _training_scope(
            select(
                EloSnapshot.fight_date,
                EloSnapshot.elo_type,
                EloSnapshot.elo_before,
            )
            .join(Fight, EloSnapshot.fight_id == Fight.id)
            .join(Event, Fight.event_id == Event.id)
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
        stmt = _training_scope(
            select(Fight.weight_class, Fight.method, Event.date).join(
                Event, Fight.event_id == Event.id
            )
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
    """Mean reach_inches across the distinct fighters who have fought in
    ``weight_class`` (training-scope fights), as
    ``feature_matrix._build_division_mean_reaches`` computes it — one entry
    per fighter, not one per fight appearance.

    Returns ``None`` when class is unknown or no fighters with known reach;
    caller treats as 0.0 → ``reach_diff_normalized`` → NaN (Pattern D).
    """
    if weight_class is None:
        return None
    try:
        from sqlalchemy import func as sa_func
        from sqlalchemy import union

        def _side(col: Any) -> Any:
            return _training_scope(
                select(col.label("fighter_id"))
                .join(Event, Fight.event_id == Event.id)
                .where(Fight.weight_class == weight_class)
            )

        fighter_ids = union(_side(Fight.fighter_a_id), _side(Fight.fighter_b_id)).subquery()
        stmt = (
            select(sa_func.avg(Fighter.reach_inches))
            .where(Fighter.id.in_(select(fighter_ids.c.fighter_id)))
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
    """Resolve a fighter's most recent transferable weight_class (best-effort).

    Same scope as training (ufcstats fights with a winner). 'Catch Weight' /
    'Open Weight' are skipped unless the fighter has fought nothing else (see
    ``_resolve_division``). For unknown / unsigned fighters returns ``None``
    → division-prior falls back to global finish rate (Bayesian fallback).
    """
    try:
        stmt = _training_scope(
            select(Fight.weight_class)
            .join(Event, Fight.event_id == Event.id)
            .where((Fight.fighter_a_id == fighter_id) | (Fight.fighter_b_id == fighter_id))
            .order_by(Event.date.desc(), Fight.id.desc())
        )
        divisions = [wc for wc in session.execute(stmt).scalars().all() if wc]
        for wc in divisions:
            if wc not in _NON_TRANSFER_DIVISIONS:
                return wc
        return divisions[0] if divisions else None
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
    weight_class: str | None = None,
) -> None:
    """Populate the 9 META cols using SAME helpers feature_matrix.py uses.

    Pitfall #12 train/predict parity: all compute via features_v22.meta.
    Pitfall #5 fix: age uses event_date (already corrected in
    ``_populate_physical``; per-fighter ``age_at_fight_*`` here also uses
    event_date).

    Only per-fighter ``layoff_days_red``/``layoff_days_blue`` are written here.
    The base-block differential ``days_since_last_fight_diff`` comes from the
    career replay in ``_populate_career`` (same accumulator as training).
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

    # Division finish rate: explicit weight class, else the fighters' most
    # recent division.
    if weight_class is None:
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


# ── Career replay (sections 5, 7-12 of the training row) ─────────────────────
#
# Mirrors the ``wc_order`` literal inside ``FeatureMatrixAssembler.assemble``
# (feature_matrix.py is AUDIT-01 frozen, so the mapping is duplicated here and
# pinned by tests/unit/ml/test_inference_career_parity.py).
_WEIGHT_CLASS_ORDINAL: dict[str, int] = {
    "Strawweight": 1,
    "Flyweight": 2,
    "Bantamweight": 3,
    "Featherweight": 4,
    "Lightweight": 5,
    "Welterweight": 6,
    "Middleweight": 7,
    "Light Heavyweight": 8,
    "Heavyweight": 9,
    "Women's Strawweight": 1,
    "Women's Flyweight": 2,
    "Women's Bantamweight": 3,
    "Women's Featherweight": 4,
}
_WEIGHT_CLASS_ORDINAL_DEFAULT = 5

_CAREER_KEYS: tuple[tuple[str, str], ...] = (
    ("win_streak", "win_streak_diff"),
    ("loss_streak", "loss_streak_diff"),
    ("career_win_pct", "career_win_pct_diff"),
    ("fight_count", "ufc_fight_count_diff"),
    ("days_since_last_fight", "days_since_last_fight_diff"),
    ("ko_finish_rate", "ko_finish_rate_diff"),
    ("sub_finish_rate", "sub_finish_rate_diff"),
    ("ko_loss_rate", "ko_loss_rate_diff"),
    ("sub_loss_rate", "sub_loss_rate_diff"),
    ("total_cage_minutes", "total_cage_minutes_diff"),
    ("avg_fight_duration", "avg_fight_duration_diff"),
    ("division_fight_count", "division_fight_count_diff"),
    ("is_debut", "is_debut_diff"),
    ("fights_per_year", "fights_per_year_diff"),
    ("avg_opponent_elo", "avg_opponent_elo_diff"),
    ("elo_momentum", "elo_momentum_diff"),
    # Section 9: non-linear layoff.
    ("log_days_since_last_fight", "log_days_since_last_fight_diff"),
    ("is_short_turnaround", "is_short_turnaround_diff"),
    ("is_comeback", "is_comeback_diff"),
    # Section 10: rolling windows.
    ("sig_str_per_min_last3", "sig_str_per_min_last3_diff"),
    ("td_rate_last3", "td_rate_last3_diff"),
    ("strike_defense_last3", "strike_defense_last3_diff"),
    ("ctrl_time_last3", "ctrl_time_last3_diff"),
    ("sig_str_per_min_last5", "sig_str_per_min_last5_diff"),
    ("td_rate_last5", "td_rate_last5_diff"),
    ("strike_defense_last5", "strike_defense_last5_diff"),
    ("ctrl_time_last5", "ctrl_time_last5_diff"),
)

_PACE_KEYS: tuple[tuple[str, str], ...] = (
    ("pace_decay_strikes", "pace_decay_strikes_diff"),
    ("pace_decay_td", "pace_decay_td_diff"),
    ("pace_output_variance", "pace_output_variance_diff"),
    ("avg_r1_sig_str", "avg_r1_sig_str_diff"),
)


@dataclass
class CareerInputs:
    """Both fighters' prior-fight substrate, in the loader shapes the
    assembler's static builders consume."""

    fight_records: list[dict[str, Any]] = field(default_factory=list)
    elo_features: dict[tuple[int, int], dict[str, float]] = field(default_factory=dict)
    computed_features: dict[tuple[int, int], dict[str, float | None]] = field(default_factory=dict)
    round_stats: dict[tuple[int, int], list[dict[str, Any]]] = field(default_factory=dict)
    pre_ufc_records: dict[int, dict[str, Any]] = field(default_factory=dict)


def _load_career_inputs(
    session: Session,
    fa_id: int,
    fb_id: int,
    before_date: date,
) -> CareerInputs:
    """Load every prior ufcstats fight (winner present, event strictly before
    ``before_date``) involving either fighter, plus the per-fight Elo,
    computed-feature and round-stat rows those fights need.

    Uses the same ``ml.queries`` loaders as training, scoped to the two
    fighters, so "a prior fight" means the same thing on both sides.
    """
    records = ml_queries.load_fight_records(
        session, fighter_ids=(fa_id, fb_id), before_date=before_date
    )
    fight_ids = [r["fight_id"] for r in records]
    if not fight_ids:
        return CareerInputs(
            pre_ufc_records=ml_queries.load_pre_ufc_records(session, fighter_ids=(fa_id, fb_id))
        )
    return CareerInputs(
        fight_records=records,
        elo_features=ml_queries.load_elo_features(session, fight_ids=fight_ids),
        computed_features=ml_queries.load_computed_features(session, fight_ids=fight_ids),
        round_stats=ml_queries.load_round_stats_for_ml(session, fight_ids=fight_ids),
        pre_ufc_records=ml_queries.load_pre_ufc_records(session, fighter_ids=(fa_id, fb_id)),
    )


def _synthetic_fight_id(records: list[dict[str, Any]]) -> int:
    """A negative id that cannot collide with any stored fight."""
    lowest = min((r["fight_id"] for r in records), default=0)
    return min(lowest, 0) - 1


def _resolve_division(
    records: list[dict[str, Any]],
    fa_id: int,
    fb_id: int,
) -> str | None:
    """Best guess at the division of an A-vs-B bout with no known weight class.

    'Catch Weight' / 'Open Weight' are one-off bout labels, not divisions: the
    Elo engine never transfers a rating into them, so resolving to one leaves
    the opponent unrated there (seed / 1500) and ``weight_class_ordinal`` on
    its default. Each fighter's most recent transferable division is a
    candidate. When they differ, prefer the one both fighters have fought in,
    else the one from the more recent fight, so the answer does not depend on
    which fighter is listed first. Falls back to the last division of any
    kind (A's, then B's) when neither has a transferable one.

    ``records`` must be chronological.
    """
    last_transferable: dict[int, tuple[date, str]] = {}
    last_any: dict[int, str] = {}
    fought: dict[int, set[str]] = {fa_id: set(), fb_id: set()}
    for r in records:
        wc = r["weight_class"]
        if not wc:
            continue
        for fid in (fa_id, fb_id):
            if fid in (r["fighter_a_id"], r["fighter_b_id"]):
                fought[fid].add(wc)
                last_any[fid] = wc
                if wc not in _NON_TRANSFER_DIVISIONS:
                    last_transferable[fid] = (r["event_date"], wc)

    cand_a = last_transferable.get(fa_id)
    cand_b = last_transferable.get(fb_id)
    if cand_a is None or cand_b is None or cand_a[1] == cand_b[1]:
        chosen = cand_a or cand_b
        if chosen is not None:
            return chosen[1]
        return last_any.get(fa_id) or last_any.get(fb_id)

    a_shared = cand_a[1] in fought[fb_id]
    b_shared = cand_b[1] in fought[fa_id]
    if a_shared != b_shared:
        return cand_a[1] if a_shared else cand_b[1]
    return cand_b[1] if cand_b[0] > cand_a[0] else cand_a[1]


def _diff(va: float, vb: float) -> float:
    if va != va or vb != vb:  # NaN on either side propagates
        return float("nan")
    return va - vb


def _populate_career(
    feats: dict[str, float],
    fa_id: int,
    fb_id: int,
    event_date: date,
    inputs: CareerInputs,
    *,
    weight_class: str | None,
    num_rounds: int,
    is_title_fight: bool,
) -> str | None:
    """Sections 5 and 7-12 of the training row (37 of the 72 base columns).

    Appends a synthetic record for the upcoming fight to both fighters'
    prior fights and runs the assembler's own static builders
    (``_build_career_stats``, ``_build_pace_stats``, ``_build_rematch_index``)
    over the sequence. The snapshot taken at the synthetic record is, by
    construction, exactly what training computes for a fight on
    ``event_date`` — the builders only ever read pre-fight state.

    Returns the weight class used (resolved by ``_resolve_division`` when not
    given) so the Elo section can replay the same division.
    """
    records = sorted(inputs.fight_records, key=lambda r: (r["event_date"], r["fight_id"]))
    if weight_class is None:
        weight_class = _resolve_division(records, fa_id, fb_id)
    upcoming_id = _synthetic_fight_id(records)
    upcoming = {
        "fight_id": upcoming_id,
        "event_date": event_date,
        "fighter_a_id": fa_id,
        "fighter_b_id": fb_id,
        "winner_id": None,
        "weight_class": weight_class or "",
        "method": None,
        "is_title_fight": is_title_fight,
        "num_rounds": num_rounds,
    }
    sequence = [*records, upcoming]

    career, fight_pairs = FeatureMatrixAssembler._build_career_stats(
        sequence, inputs.elo_features, inputs.computed_features
    )
    pace = FeatureMatrixAssembler._build_pace_stats(sequence, inputs.round_stats)
    rematch = FeatureMatrixAssembler._build_rematch_index(fight_pairs)

    career_a = career.get((fa_id, upcoming_id), {})
    career_b = career.get((fb_id, upcoming_id), {})
    for src, dst in _CAREER_KEYS:
        feats[dst] = _diff(career_a.get(src, math.nan), career_b.get(src, math.nan))

    # Section 7: context flags (never NaN in training).
    feats["is_title_fight"] = 1.0 if is_title_fight else 0.0
    feats["num_rounds"] = float(num_rounds)
    feats["weight_class_ordinal"] = float(
        _WEIGHT_CLASS_ORDINAL.get(weight_class or "", _WEIGHT_CLASS_ORDINAL_DEFAULT)
    )

    # Section 8: pace decay.
    pace_a = pace.get((fa_id, upcoming_id), {})
    pace_b = pace.get((fb_id, upcoming_id), {})
    for src, dst in _PACE_KEYS:
        feats[dst] = _diff(pace_a.get(src, math.nan), pace_b.get(src, math.nan))

    # Section 11: rematch.
    rm = rematch.get(upcoming_id, {})
    feats["is_rematch"] = float(rm.get("is_rematch", 0))
    first_winner = rm.get("first_fight_winner")
    if first_winner == fa_id:
        feats["first_fight_winner_diff"] = 1.0
    elif first_winner == fb_id:
        feats["first_fight_winner_diff"] = -1.0
    else:
        feats["first_fight_winner_diff"] = 0.0

    # Section 12: pre-UFC record.
    pre_a = inputs.pre_ufc_records.get(fa_id) or {}
    pre_b = inputs.pre_ufc_records.get(fb_id) or {}
    feats["pre_ufc_win_pct_diff"] = _diff(
        pre_a.get("win_pct", math.nan) if pre_a else math.nan,
        pre_b.get("win_pct", math.nan) if pre_b else math.nan,
    )
    return weight_class


def _populate_physical(
    fighter_a,
    fighter_b,
    feats: dict[str, float],
    event_date: date,
    *,
    session: Session | None = None,
    weight_class: str | None = None,
) -> None:
    """Section 3: Physical differentials (4 features) + stance (1 feature).

    Reads attributes directly off the Fighter ORM rows (already loaded by
    the predictor's ``_resolve_fighter`` call). Mirrors training's D-01
    imputation (``feature_matrix.py`` section 3): a missing height / reach /
    leg reach is replaced by the division median computed with the training
    cutoff before differencing; the difference is NaN only when the division
    has no median either. Medians are read only when a value is missing.

    Phase 23 Pitfall #5 fix: ``age_diff`` is computed from ``event_date``
    (NOT ``date.today()``) so backtests against historical events compute
    the correct age. The age at a historical fight is determined by the
    event_date, not by the current calendar date. Prior to this fix, the
    inference path diverged from the training path
    (``feature_matrix.py:650-659`` already uses ``event_date``) for any
    ``event_date != today`` — a silent parity break for OOF generation.
    """
    div_med: dict[str, float] | None = None
    for attr, key in (
        ("height_inches", "height_diff"),
        ("reach_inches", "reach_diff"),
        ("leg_reach_inches", "leg_reach_diff"),
    ):
        val_a = getattr(fighter_a, attr, None)
        val_b = getattr(fighter_b, attr, None)
        if (val_a is None or val_b is None) and session is not None:
            if div_med is None:
                div_med = _query_division_physical_medians(
                    session, weight_class, date.fromisoformat(MLConfig().cutoff_date)
                )
            median_val = div_med.get(attr)
            if val_a is None:
                val_a = median_val
            if val_b is None:
                val_b = median_val
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
        # Live BFO path: bfo_live fills A from fighter A's profile row and B
        # from the opponent row of the same fight, and only returns a live
        # hit when at least one two-sided (opening or closing) pair exists.
        # Any individual side still missing must NaN out per Pattern D.
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
    weight_class: str | None = None,
    num_rounds: int | None = None,
    is_title_fight: bool | None = None,
) -> np.ndarray:
    """Build a ``(1, len(cols))`` feature vector from the fighters' stored
    history as of ``event_date`` + injected live/cached odds.

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
        event_date: As-of date of the fight. Every section reads only
            history strictly before it (performance / NET snapshots are
            replayed as of this date, so a historical ``event_date`` never
            sees later fights) and it keys the cached-odds lookup.
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
        weight_class: Division of the upcoming fight. Drives
            ``weight_class_ordinal``, ``division_fight_count_diff``, the
            per-division Elo replay and the physical-median imputation.
        num_rounds: Scheduled rounds (3 or 5) → ``num_rounds`` column.
        is_title_fight: → ``is_title_fight`` column.

            Any of the three left ``None`` is read from the stored ``Fight``
            row for ``(A, B, event_date)`` when one exists (the same row
            training reads). Without one, ``num_rounds`` defaults to 3,
            ``is_title_fight`` to False and ``weight_class`` to
            ``_resolve_division`` over both fighters' prior fights.

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

    # Bout context: explicit args win, then the stored Fight row for this
    # matchup and date, then the defaults / division fallback.
    if weight_class is None or num_rounds is None or is_title_fight is None:
        scheduled = _query_scheduled_bout(session, fighter_a.id, fighter_b.id, event_date)
        if scheduled is not None:
            sched_wc, sched_rounds, sched_title = scheduled
            weight_class = weight_class if weight_class is not None else sched_wc
            num_rounds = num_rounds if num_rounds is not None else sched_rounds
            is_title_fight = is_title_fight if is_title_fight is not None else sched_title
    if num_rounds is None:
        num_rounds = 3
    if is_title_fight is None:
        is_title_fight = False

    # Sections 5, 7-12: career / context / pace / layoff / rolling / rematch /
    # pre-UFC — replayed from both fighters' prior fights with the assembler's
    # own builders. Runs first so the resolved division feeds the Elo replay.
    career_inputs = _load_career_inputs(session, fighter_a.id, fighter_b.id, event_date)
    weight_class = _populate_career(
        feats,
        fighter_a.id,
        fighter_b.id,
        event_date,
        career_inputs,
        weight_class=weight_class,
        num_rounds=num_rounds,
        is_title_fight=is_title_fight,
    )

    # Section 1 & 2: Elo + performance differentials
    elo_a_overall, elo_b_overall = _populate_elo(
        session,
        fighter_a.id,
        fighter_b.id,
        feats,
        event_date,
        weight_class,
    )
    feats_a_latest, feats_b_latest = _populate_performance(
        session,
        fighter_a.id,
        fighter_b.id,
        feats,
        event_date,
    )

    # Section 3: Physical + stance (Pitfall #5 fix — age uses event_date)
    _populate_physical(
        fighter_a, fighter_b, feats, event_date, session=session, weight_class=weight_class
    )

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
    # Reuses the same pre-fight snapshots _populate_performance replayed —
    # NET-* keys come from FeatureComputer Pass 4 on that snapshot.
    # Train/predict parity (Pitfall #12) is enforced by sharing
    # compute_network_diff_features with feature_matrix.py.
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
        _populate_meta(session, feats, fighter_a, fighter_b, event_date, weight_class)

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
