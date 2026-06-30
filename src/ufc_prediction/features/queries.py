"""Database query layer for feature computation.

Batch queries for loading round stats, fights, domain Elo, and
storing computed features. Follows the pattern from elo/queries.py.
All queries use SQLAlchemy parameterized select() statements (T-07-02).
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ufc_prediction.models.computed_feature import ComputedFeature
from ufc_prediction.models.elo_snapshot import EloSnapshot
from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.round_stats import RoundStats


def load_all_round_stats(session: Session) -> dict[int, list[dict[str, object]]]:
    """Load ALL RoundStats columns indexed by fight_id for feature computation.

    Returns dict mapping fight_id -> list of per-round dicts with all stat
    columns preserved as-is (None or int) so rate computation can detect
    missing data.

    Single batch query, O(1) lookup per fight. Ordered by fight_id, round_number.
    """
    stmt = select(
        RoundStats.fight_id,
        RoundStats.fighter_id,
        RoundStats.round_number,
        RoundStats.sig_strikes_landed,
        RoundStats.sig_strikes_attempted,
        RoundStats.head_strikes_landed,
        RoundStats.body_strikes_landed,
        RoundStats.leg_strikes_landed,
        RoundStats.distance_strikes_landed,
        RoundStats.clinch_strikes_landed,
        RoundStats.ground_strikes_landed,
        RoundStats.takedowns_landed,
        RoundStats.takedowns_attempted,
        RoundStats.submission_attempts,
        RoundStats.reversals,
        RoundStats.control_time_seconds,
        RoundStats.knockdowns,
    ).order_by(RoundStats.fight_id, RoundStats.round_number)

    rows = session.execute(stmt).all()
    result: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        fight_id = row[0]
        if fight_id not in result:
            result[fight_id] = []
        result[fight_id].append(
            {
                "fighter_id": row[1],
                "round_number": row[2],
                "sig_strikes_landed": row[3],
                "sig_strikes_attempted": row[4],
                "head_strikes_landed": row[5],
                "body_strikes_landed": row[6],
                "leg_strikes_landed": row[7],
                "distance_strikes_landed": row[8],
                "clinch_strikes_landed": row[9],
                "ground_strikes_landed": row[10],
                "takedowns_landed": row[11],
                "takedowns_attempted": row[12],
                "submission_attempts": row[13],
                "reversals": row[14],
                "control_time_seconds": row[15],
                "knockdowns": row[16],
            }
        )
    return result


def load_fights_with_duration(session: Session) -> list[dict[str, object]]:
    """Load fights joined with events for chronological ordering.

    Returns list of dicts ordered by event date, then fight ID
    (deterministic within same date, same pattern as elo/queries.py).
    """
    stmt = (
        select(
            Fight.id,
            Event.date,
            Fight.fighter_a_id,
            Fight.fighter_b_id,
            Fight.weight_class,
            Fight.round_finished,
            Fight.time_finished,
            Fight.num_rounds,
        )
        .join(Event, Fight.event_id == Event.id)
        .order_by(Event.date, Fight.id)
    )

    rows = session.execute(stmt).all()
    return [
        {
            "fight_id": row[0],
            "event_date": row[1],
            "fighter_a_id": row[2],
            "fighter_b_id": row[3],
            "weight_class": row[4],
            "round_finished": row[5],
            "time_finished": row[6],
            "num_rounds": row[7],
        }
        for row in rows
    ]


def load_all_domain_elo(
    session: Session,
) -> dict[tuple[int, int], dict[str, float | None]]:
    """Load domain Elo (striking + grappling) for every fighter/fight combination.

    Single batch query instead of per-fighter lookups (avoids N+1).
    Returns dict mapping (fighter_id, fight_id) -> {
        "striking_elo": float | None,
        "grappling_elo": float | None,
    }.
    """
    stmt = select(
        EloSnapshot.fighter_id,
        EloSnapshot.fight_id,
        EloSnapshot.elo_type,
        EloSnapshot.elo_after_shrinkage,
    ).where(
        EloSnapshot.elo_type.in_(["striking", "grappling"]),
    )

    rows = session.execute(stmt).all()
    result: dict[tuple[int, int], dict[str, float | None]] = {}
    for row in rows:
        key = (row[0], row[1])
        if key not in result:
            result[key] = {"striking_elo": None, "grappling_elo": None}
        if row[2] == "striking":
            result[key]["striking_elo"] = row[3]
        else:
            result[key]["grappling_elo"] = row[3]
    return result


def flush_features(session: Session, version: str = "v1") -> None:
    """Delete all ComputedFeature rows with the given version (idempotent).

    Per D-15: recomputing overwrites existing feature rows.
    Same pattern as flush_snapshots in elo/queries.py.

    Does NOT commit -- caller is responsible for transaction management.
    """
    session.query(ComputedFeature).filter(
        ComputedFeature.feature_set_version == version,
    ).delete(synchronize_session=False)
    session.flush()


def bulk_insert_features(session: Session, feature_rows: list[dict[str, object]]) -> int:
    """Bulk insert computed feature rows.

    Each dict must have: fighter_id, fight_id, as_of_date,
    feature_set_version, features (JSON dict).

    Returns count of rows inserted.
    Does NOT commit -- caller is responsible for transaction management.
    """
    if not feature_rows:
        return 0

    session.execute(ComputedFeature.__table__.insert(), feature_rows)  # type: ignore[attr-defined]
    return len(feature_rows)
