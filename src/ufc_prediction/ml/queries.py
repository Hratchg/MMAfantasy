"""Database query layer for ML feature matrix assembly.

Batch queries for loading Elo features, computed features, fighter physicals,
and fight records. Follows the pattern from features/queries.py.
All queries use SQLAlchemy parameterized select() statements (T-10-03).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ufc_prediction.ml.config import PERFORMANCE_FEATURE_KEYS
from ufc_prediction.models.computed_feature import ComputedFeature
from ufc_prediction.models.elo_snapshot import EloSnapshot
from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fighter import Fighter
from ufc_prediction.models.round_stats import RoundStats


def load_elo_features(session: Session) -> dict[tuple[int, int], dict[str, float]]:
    """Load Elo features indexed by (fighter_id, fight_id).

    Returns elo_before for overall, striking, grappling.
    CRITICAL: Uses elo_before (NOT elo_after) to avoid data leakage (T-10-01).
    Default to 1500.0 for missing elo types.
    """
    stmt = select(
        EloSnapshot.fighter_id,
        EloSnapshot.fight_id,
        EloSnapshot.elo_type,
        EloSnapshot.elo_before,
    )

    rows = session.execute(stmt).all()
    result: dict[tuple[int, int], dict[str, float]] = {}
    for row in rows:
        key = (row[0], row[1])
        if key not in result:
            result[key] = {
                "elo_overall": 1500.0,
                "elo_striking": 1500.0,
                "elo_grappling": 1500.0,
            }
        result[key][f"elo_{row[2]}"] = row[3]
    return result


def load_computed_features(
    session: Session,
) -> dict[tuple[int, int], dict[str, float | None]]:
    """Load computed features indexed by (fighter_id, fight_id).

    Extracts the 20 numeric features from CANONICAL_FEATURE_ORDER
    (excluding style_tag). None values are preserved (XGBoost handles NaN).

    TEMPORAL-V24-01: rows are filtered such that the feature's ``as_of_date``
    is not later than the joined ``Event.date``. Future-dated rows
    (operator-bug temporal leaks) are excluded from the training + inference
    feature pipeline at this single chokepoint. NULL ``as_of_date`` rows
    (pre-Phase-37 legacy data) are conservatively INCLUDED to preserve
    historical loader behavior — only rows with an explicit ``as_of_date``
    strictly after the event are dropped. See
    ``tests/regression/test_computed_features_temporal.py`` (TEMPORAL-V24-02)
    for the regression test pinning this contract.
    """
    stmt = (
        select(
            ComputedFeature.fighter_id,
            ComputedFeature.fight_id,
            ComputedFeature.features,
        )
        .join(Fight, ComputedFeature.fight_id == Fight.id)
        .join(Event, Fight.event_id == Event.id)
        .where(
            # TEMPORAL-V24-01 — code-enforce temporal validity: a computed
            # feature's as_of_date may never be later than the event it
            # describes. Future-dated rows (operator-bug leaks) are excluded
            # from the training + inference feature pipeline. NULL handling
            # is conservative: pre-Phase-37 legacy rows with no as_of_date
            # are INCLUDED. See tests/regression/test_computed_features_temporal.py.
            (ComputedFeature.as_of_date.is_(None)) | (ComputedFeature.as_of_date <= Event.date)
        )
    )

    rows = session.execute(stmt).all()
    result: dict[tuple[int, int], dict[str, float | None]] = {}
    for row in rows:
        key = (row[0], row[1])
        features_json = row[2]
        result[key] = {feat: features_json.get(feat) for feat in PERFORMANCE_FEATURE_KEYS}
    return result


def load_fighter_physicals(session: Session) -> dict[int, dict[str, Any]]:
    """Load fighter physical attributes indexed by fighter_id.

    Returns raw values; imputation happens in the assembler (D-01).
    """
    stmt = select(
        Fighter.id,
        Fighter.height_inches,
        Fighter.reach_inches,
        Fighter.leg_reach_inches,
        Fighter.stance,
        Fighter.date_of_birth,
    )

    rows = session.execute(stmt).all()
    return {
        row[0]: {
            "height_inches": row[1],
            "reach_inches": row[2],
            "leg_reach_inches": row[3],
            "stance": row[4],
            "date_of_birth": row[5],
        }
        for row in rows
    }


def load_fight_records(session: Session) -> list[dict[str, Any]]:
    """Load fight records joined with events for chronological ordering.

    Returns list of dicts ordered by event_date ASC, fight_id ASC.
    Skips fights where winner_id is None (draws/no-contests) since
    they have no prediction target (per existing backtesting.py pattern).

    Plan 28-04 Task 1 (Path B at load) — closes the HIGH-severity finding
    documented in
    ``.planning/phases/28-referee-venue-ingestion-pipeline/28-FEATURE-MATRIX-DEDUP-FINDING.md``:
    the previous query had no ``Event.source`` predicate, so it returned
    cross-source duplicates (ufcstats + kaggle-mdabbert + kaggle-rajeevw)
    that inflated the v2.x training corpus by ~1.88×. The
    ``.where(Event.source == 'ufcstats')`` clause filters to the canonical
    ufcstats baseline (~8,473 fights with winner_id, per
    ``28-DEDUP-RECON.md`` Section 1). Path A (canonical re-link/delete) is
    deferred per ``28-DEDUP-RECON.md`` Section 6.

    Defensive note: equality predicate on ``source`` drops NULL rows under
    SQL three-valued logic — currently no NULLs exist in production, but
    the filter holds the contract if future ingestion bugs introduce them.
    """
    stmt = (
        select(
            Fight.id,
            Event.date,
            Fight.fighter_a_id,
            Fight.fighter_b_id,
            Fight.winner_id,
            Fight.weight_class,
            Fight.method,
            Fight.is_title_fight,
            Fight.num_rounds,
        )
        .join(Event, Fight.event_id == Event.id)
        .where(Fight.winner_id.isnot(None))
        .where(Event.source == "ufcstats")
        .order_by(Event.date, Fight.id)
    )

    rows = session.execute(stmt).all()
    return [
        {
            "fight_id": row[0],
            "event_date": row[1],
            "fighter_a_id": row[2],
            "fighter_b_id": row[3],
            "winner_id": row[4],
            "weight_class": row[5],
            "method": row[6],
            "is_title_fight": row[7],
            "num_rounds": row[8],
        }
        for row in rows
    ]


def load_pre_ufc_records(session: Session) -> dict[int, dict[str, Any]]:
    """Load pre-UFC career records indexed by fighter_id.

    Returns dict mapping fighter_id -> pre_ufc_record JSON dict
    for fighters that have Sherdog-sourced pre-UFC data.
    Keys in the JSON: total_wins, total_losses, total_fights, win_pct,
    ko_finish_rate, sub_finish_rate, etc.
    """
    stmt = select(Fighter.id, Fighter.pre_ufc_record).where(Fighter.pre_ufc_record.isnot(None))

    rows = session.execute(stmt).all()
    return {row[0]: row[1] for row in rows}


def load_round_stats_for_ml(
    session: Session,
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    """Load round stats indexed by (fighter_id, fight_id) for pace decay features.

    Returns dict mapping (fighter_id, fight_id) -> list of per-round dicts
    sorted by round_number. Each dict has: round_number, sig_str_landed, td_landed.
    """
    stmt = (
        select(
            RoundStats.fighter_id,
            RoundStats.fight_id,
            RoundStats.round_number,
            RoundStats.sig_strikes_landed,
            RoundStats.takedowns_landed,
        )
        .where(RoundStats.round_number > 0)  # skip round 0 aggregates
        .order_by(RoundStats.fight_id, RoundStats.fighter_id, RoundStats.round_number)
    )

    rows = session.execute(stmt).all()
    result: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row[0], row[1])
        if key not in result:
            result[key] = []
        result[key].append(
            {
                "round_number": row[2],
                "sig_str_landed": row[3] or 0,
                "td_landed": row[4] or 0,
            }
        )
    return result


def load_fight_odds(
    session: Session,
) -> dict[tuple[int, int], dict[str, float | None]]:
    """Load implied-probability odds indexed by (fighter_id, fight_id).

    Returns only the columns the ML feature matrix needs:
      - opening_implied_prob (vig-removed per D-02)
      - closing_implied_prob (vig-removed midpoint per D-02 + D-01)

    Raw American moneyline columns (opening_ml, closing_range_min_ml,
    closing_range_max_ml) are stored in the DB (D-03) but not returned
    here — they are preserved for audit/future use but are not model
    features.

    Missing fighters/fights are simply absent from the dict; the feature
    matrix assembler interprets absence as NaN (D-07). None values for
    a present row are preserved (XGBoost-side NaN coercion happens in
    the assembler, not here — Pitfall 3).
    """
    # Inline import keeps module-import-time coupling low and matches the
    # defensive pattern used in scraper-adjacent paths where FightOdds
    # could otherwise create a circular dependency at fixture setup.
    from ufc_prediction.models.fight_odds import FightOdds

    stmt = select(
        FightOdds.fighter_id,
        FightOdds.fight_id,
        FightOdds.opening_implied_prob,
        FightOdds.closing_implied_prob,
    )
    rows = session.execute(stmt).all()
    return {
        (row[0], row[1]): {
            "opening_implied_prob": row[2],
            "closing_implied_prob": row[3],
        }
        for row in rows
    }
