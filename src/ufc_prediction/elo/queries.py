"""Database query layer for Elo computation.

Loads fights chronologically from the database and flushes computed
Elo snapshots back to the elo_snapshots table.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ufc_prediction.elo.engine import FightRecord, SnapshotRecord
from ufc_prediction.models.elo_snapshot import EloSnapshot
from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.round_stats import RoundStats


def load_fights_chronological(session: Session) -> list[FightRecord]:
    """Load all fights from DB ordered by event date, then fight ID.

    Deterministic ordering within same-date events per RESEARCH.md Pitfall 5.
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
            Fight.method_detail,
        )
        .join(Event, Fight.event_id == Event.id)
        .order_by(Event.date, Fight.id)
    )
    rows = session.execute(stmt).all()
    return [
        FightRecord(
            fight_id=row[0],
            event_date=row[1],
            fighter_a_id=row[2],
            fighter_b_id=row[3],
            winner_id=row[4],
            weight_class=row[5],
            method=row[6],
            method_detail=row[7],
        )
        for row in rows
    ]


def flush_snapshots(session: Session, snapshots: list[SnapshotRecord]) -> int:
    """Delete existing overall snapshots and bulk-insert new ones (idempotent).

    Scoped to elo_type == "overall" only (T-03-07: won't affect future
    domain-specific snapshots from Phase 6).

    Does NOT commit -- caller is responsible for transaction management.
    """
    session.query(EloSnapshot).filter(EloSnapshot.elo_type == "overall").delete()
    session.flush()

    if not snapshots:
        return 0

    dicts = [
        {
            "fighter_id": s.fighter_id,
            "fight_id": s.fight_id,
            "division": s.division,
            "elo_type": s.elo_type,
            "elo_before": s.elo_before,
            "elo_after": s.elo_after,
            "elo_after_shrinkage": s.elo_after_shrinkage,
            "k_factor_used": s.k_factor_used,
            "fight_date": s.fight_date,
        }
        for s in snapshots
    ]
    session.execute(EloSnapshot.__table__.insert(), dicts)  # type: ignore[attr-defined]
    return len(snapshots)


def load_round_stats_by_fight(session: Session) -> dict[int, list[dict[str, object]]]:
    """Load all round stats indexed by fight_id for batch domain Elo computation.

    Returns dict mapping fight_id -> list of dicts with keys:
    fighter_id, sig_str_landed, takedowns_landed, ctrl_time_seconds.

    Single query, O(1) lookup per fight. Avoids N+1 query pitfall (T-06-02).
    """
    stmt = select(
        RoundStats.fight_id,
        RoundStats.fighter_id,
        RoundStats.sig_strikes_landed,
        RoundStats.takedowns_landed,
        RoundStats.control_time_seconds,
    ).order_by(RoundStats.fight_id, RoundStats.round_number)

    rows = session.execute(stmt).all()
    result: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        fight_id = row[0]
        if fight_id not in result:
            result[fight_id] = []
        result[fight_id].append({
            "fighter_id": row[1],
            "sig_str_landed": row[2] or 0,
            "takedowns_landed": row[3] or 0,
            "ctrl_time_seconds": row[4] or 0,
        })
    return result


def flush_domain_snapshots(
    session: Session, snapshots: list[SnapshotRecord],
) -> int:
    """Delete existing domain snapshots and bulk-insert new ones (idempotent).

    Scoped to elo_type in ("striking", "grappling") only.
    Does NOT affect overall snapshots (T-03-07 preserved, T-06-01 mitigated).

    Does NOT commit -- caller is responsible for transaction management.
    """
    session.query(EloSnapshot).filter(
        EloSnapshot.elo_type.in_(["striking", "grappling"]),
    ).delete(synchronize_session=False)
    session.flush()

    if not snapshots:
        return 0

    dicts = [
        {
            "fighter_id": s.fighter_id,
            "fight_id": s.fight_id,
            "division": s.division,
            "elo_type": s.elo_type,
            "elo_before": s.elo_before,
            "elo_after": s.elo_after,
            "elo_after_shrinkage": s.elo_after_shrinkage,
            "k_factor_used": s.k_factor_used,
            "fight_date": s.fight_date,
        }
        for s in snapshots
    ]
    session.execute(EloSnapshot.__table__.insert(), dicts)  # type: ignore[attr-defined]
    return len(snapshots)
