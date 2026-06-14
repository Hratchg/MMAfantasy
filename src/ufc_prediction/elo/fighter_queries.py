"""Query functions for fighter lookup and division rankings.

Provides read-only query functions used by the CLI fighter commands.
All functions accept a SQLAlchemy Session and return plain Python data
structures (no Rich/Typer dependencies). Uses SQLAlchemy 2.0 select() style.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ufc_prediction.dedup.source_priority import prefer_canonical
from ufc_prediction.models.elo_snapshot import EloSnapshot
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fighter import Fighter, FighterAlias

# Phase 16 HOUSE-04: the local `_RANKINGS_SOURCE_PRIORITY` dict was unified
# into `ufc_prediction.dedup.source_priority.SOURCE_PRIORITY`. The
# get_division_rankings call site below now uses `prefer_canonical(...,
# tiebreak_key=lambda r: -float(r.elo_after_shrinkage))` to preserve the
# highest-Elo-wins-within-same-source contract from Phase 14 (DEDUP-03).
# The unified module sidesteps the original elo/ -> ml/ import concern.

# All 14 valid UFC weight classes (defined locally per RESEARCH anti-pattern
# advice to avoid coupling elo module to data.schemas)
_ALL_DIVISIONS: frozenset[str] = frozenset({
    "Flyweight",
    "Bantamweight",
    "Featherweight",
    "Lightweight",
    "Welterweight",
    "Middleweight",
    "Light Heavyweight",
    "Heavyweight",
    "Women's Strawweight",
    "Women's Flyweight",
    "Women's Bantamweight",
    "Women's Featherweight",
    "Catch Weight",
    "Open Weight",
})

# Rankable divisions exclude Catch Weight and Open Weight.
# Sorted longest-first for greedy substring matching.
_RANKABLE_DIVISIONS: list[str] = sorted(
    [wc for wc in _ALL_DIVISIONS if wc not in {"Catch Weight", "Open Weight"}],
    key=lambda x: -len(x),
)


def search_fighters(session: Session, name: str) -> list[Fighter]:
    """Search fighters by name or alias (case-insensitive ILIKE).

    Uses parameterized queries via SQLAlchemy bind parameters (T-04-01).
    """
    pattern = f"%{name}%"
    stmt = (
        select(Fighter)
        .outerjoin(FighterAlias, Fighter.id == FighterAlias.fighter_id)
        .where(
            or_(
                Fighter.name.ilike(pattern),
                FighterAlias.alias_name.ilike(pattern),
            )
        )
        .distinct()
    )
    return list(session.scalars(stmt).all())


def get_fighter_detail(
    session: Session,
    fighter_id: int,
    division: str | None = None,
) -> dict[str, Any]:
    """Get fighter detail including Elo, win/loss record, and last fight date.

    When division is None, returns the most recent snapshot across all divisions.
    """
    # Query the most recent EloSnapshot
    snap_stmt = (
        select(EloSnapshot)
        .where(EloSnapshot.fighter_id == fighter_id)
        .where(EloSnapshot.elo_type == "overall")
    )
    if division is not None:
        snap_stmt = snap_stmt.where(EloSnapshot.division == division)

    snap_stmt = snap_stmt.order_by(EloSnapshot.fight_date.desc()).limit(1)
    snapshot = session.scalars(snap_stmt).first()

    if snapshot is None:
        return {
            "elo": None,
            "division": division,
            "wins": 0,
            "losses": 0,
            "total_fights": 0,
            "last_fight_date": None,
        }

    effective_division = snapshot.division

    # Count wins and losses in this division
    fight_filter = or_(
        Fight.fighter_a_id == fighter_id,
        Fight.fighter_b_id == fighter_id,
    )
    if division is not None:
        fight_filter = fight_filter & (Fight.weight_class == division)
    else:
        fight_filter = fight_filter & (Fight.weight_class == effective_division)

    total_stmt = select(func.count()).select_from(Fight).where(fight_filter)
    total_fights = session.scalar(total_stmt) or 0

    wins_stmt = (
        select(func.count())
        .select_from(Fight)
        .where(fight_filter)
        .where(Fight.winner_id == fighter_id)
    )
    wins = session.scalar(wins_stmt) or 0

    # Count losses explicitly — exclude draws/no-contests (winner_id IS NULL)
    losses_stmt = (
        select(func.count())
        .select_from(Fight)
        .where(fight_filter)
        .where(Fight.winner_id != None)  # noqa: E711
        .where(Fight.winner_id != fighter_id)
    )
    losses = session.scalar(losses_stmt) or 0

    return {
        "elo": snapshot.elo_after_shrinkage,
        "division": effective_division,
        "wins": wins,
        "losses": losses,
        "total_fights": total_fights,
        "last_fight_date": snapshot.fight_date,
    }


def get_fighter_divisions(session: Session, fighter_id: int) -> list[str]:
    """Get all divisions a fighter has competed in (from Elo snapshots)."""
    stmt = (
        select(EloSnapshot.division)
        .where(EloSnapshot.fighter_id == fighter_id)
        .where(EloSnapshot.elo_type == "overall")
        .distinct()
    )
    return sorted(session.scalars(stmt).all())


def get_division_rankings(
    session: Session,
    division: str,
    limit: int = 15,
) -> list[dict[str, Any]]:
    """Get top fighters in a division ranked by Elo (elo_after_shrinkage desc).

    Uses only the most recent snapshot per fighter to avoid duplicates.
    """
    # Subquery: latest fight_date per fighter in division
    latest = (
        select(
            EloSnapshot.fighter_id,
            func.max(EloSnapshot.fight_date).label("max_date"),
        )
        .where(EloSnapshot.elo_type == "overall")
        .where(EloSnapshot.division == division)
        .group_by(EloSnapshot.fighter_id)
        .subquery()
    )

    # Count fights per fighter in division (number of elo snapshots)
    fight_counts = (
        select(
            EloSnapshot.fighter_id,
            func.count().label("fight_count"),
        )
        .where(EloSnapshot.elo_type == "overall")
        .where(EloSnapshot.division == division)
        .group_by(EloSnapshot.fighter_id)
        .subquery()
    )

    # Main query: join snapshots with latest dates and fighter names.
    # DEDUP-03 (Phase 14): include Fighter.source so we can dedup duplicate
    # fighter rows (same real-world person across ingest sources) by source
    # priority before truncating to `limit`. Over-fetch to limit*2 so
    # post-dedup truncation cannot drop a real fighter from the top-N when
    # duplicate pairs sit just above the cutoff.
    stmt = (
        select(
            EloSnapshot.elo_after_shrinkage,
            EloSnapshot.fight_date,
            EloSnapshot.fighter_id,
            Fighter.name,
            Fighter.source,
            fight_counts.c.fight_count,
        )
        .join(
            latest,
            (EloSnapshot.fighter_id == latest.c.fighter_id)
            & (EloSnapshot.fight_date == latest.c.max_date),
        )
        .join(Fighter, Fighter.id == EloSnapshot.fighter_id)
        .join(fight_counts, fight_counts.c.fighter_id == EloSnapshot.fighter_id)
        .where(EloSnapshot.elo_type == "overall")
        .where(EloSnapshot.division == division)
        .order_by(EloSnapshot.elo_after_shrinkage.desc())
        .limit(limit * 2)
    )

    rows = session.execute(stmt).all()

    # DEDUP-03: rows are already sorted by elo_after_shrinkage DESC. Group by
    # name, then pick the canonical (highest-source-priority) row per name.
    # Within the same name+source priority, prefer the higher Elo entry.
    by_name: dict[str, list[Any]] = {}
    for row in rows:
        by_name.setdefault(row.name.lower(), []).append(row)

    # Phase 16 HOUSE-04: replaces the local _RANKINGS_SOURCE_PRIORITY +
    # inline min() call site. tiebreak_key=-elo_after_shrinkage preserves
    # the "highest Elo wins within same source" semantic (Gotcha 3).
    canonical_rows = [
        prefer_canonical(
            group,
            source_key=lambda r: r.source,
            tiebreak_key=lambda r: -float(r.elo_after_shrinkage),
        )
        for group in by_name.values()
    ]

    # Re-sort by Elo desc since dict insertion order may not match Elo order
    # after dedup, then truncate to the original limit.
    canonical_rows.sort(key=lambda r: r.elo_after_shrinkage, reverse=True)
    canonical_rows = canonical_rows[:limit]

    return [
        {
            "name": row.name,
            "elo": row.elo_after_shrinkage,
            "fighter_id": row.fighter_id,
            "fights": row.fight_count,
            "last_date": row.fight_date,
        }
        for row in canonical_rows
    ]


def resolve_weight_class(input_str: str) -> list[str]:
    """Resolve partial weight class input to matching divisions.

    Excludes Catch Weight and Open Weight from results.
    Case-insensitive substring matching.
    """
    needle = input_str.lower()
    return [wc for wc in _RANKABLE_DIVISIONS if needle in wc.lower()]


def get_fighter_domain_elo(
    session: Session,
    fighter_id: int,
    division: str,
) -> dict[str, float | None]:
    """Get a fighter's most recent domain Elo ratings in a division.

    Returns {"striking_elo": float|None, "grappling_elo": float|None}.
    Returns None values when no domain snapshots exist for the fighter/division.

    Uses parameterized queries via SQLAlchemy bind parameters (T-06-05).
    """
    result: dict[str, float | None] = {}
    for elo_type in ("striking", "grappling"):
        stmt = (
            select(EloSnapshot.elo_after_shrinkage)
            .where(EloSnapshot.fighter_id == fighter_id)
            .where(EloSnapshot.division == division)
            .where(EloSnapshot.elo_type == elo_type)
            .order_by(EloSnapshot.fight_date.desc())
            .limit(1)
        )
        value = session.scalar(stmt)
        result[f"{elo_type}_elo"] = value
    return result


def list_divisions() -> list[str]:
    """Return all rankable divisions (excludes Catch Weight and Open Weight)."""
    return list(_RANKABLE_DIVISIONS)


def get_all_fighters_with_ratings(session: Session) -> list[dict[str, Any]]:
    """Get all fighters with their latest overall Elo, plus domain Elo.

    Returns a flat list of dicts, one row per fighter per division.
    Efficient single-query approach for the main join (no N+1).
    Domain Elo uses per-row lookup (acceptable for offline CLI export).
    """
    # Subquery: latest fight_date per fighter per division for overall type
    latest = (
        select(
            EloSnapshot.fighter_id,
            EloSnapshot.division,
            func.max(EloSnapshot.fight_date).label("max_date"),
        )
        .where(EloSnapshot.elo_type == "overall")
        .group_by(EloSnapshot.fighter_id, EloSnapshot.division)
        .subquery()
    )

    # Main: join fighter + latest snapshot
    stmt = (
        select(
            Fighter.name,
            Fighter.id,
            EloSnapshot.division,
            EloSnapshot.elo_after_shrinkage.label("elo"),
            EloSnapshot.fight_date.label("last_fight_date"),
        )
        .join(
            latest,
            (EloSnapshot.fighter_id == latest.c.fighter_id)
            & (EloSnapshot.division == latest.c.division)
            & (EloSnapshot.fight_date == latest.c.max_date),
        )
        .join(Fighter, Fighter.id == EloSnapshot.fighter_id)
        .where(EloSnapshot.elo_type == "overall")
        .order_by(Fighter.name, EloSnapshot.division)
    )

    rows = session.execute(stmt).all()
    results = []
    for row in rows:
        domain = get_fighter_domain_elo(session, row.id, row.division)
        results.append({
            "name": row.name,
            "fighter_id": row.id,
            "division": row.division,
            "elo": row.elo,
            "striking_elo": domain["striking_elo"],
            "grappling_elo": domain["grappling_elo"],
            "last_fight_date": str(row.last_fight_date) if row.last_fight_date else "",
        })
    return results


def get_elo_history(
    session: Session,
    fighter_id: int,
    division: str | None = None,
    elo_type: str = "overall",
) -> list[dict[str, Any]]:
    """Get chronological Elo history for a fighter.

    Returns list of dicts with fight_date, division, elo_before, elo_after,
    and elo_after_shrinkage for each fight.
    """
    stmt = (
        select(EloSnapshot)
        .where(EloSnapshot.fighter_id == fighter_id)
        .where(EloSnapshot.elo_type == elo_type)
    )
    if division is not None:
        stmt = stmt.where(EloSnapshot.division == division)
    stmt = stmt.order_by(EloSnapshot.fight_date.asc())

    snapshots = session.scalars(stmt).all()
    return [
        {
            "fight_date": s.fight_date,
            "division": s.division,
            "elo_before": s.elo_before,
            "elo_after": s.elo_after,
            "elo_after_shrinkage": s.elo_after_shrinkage,
        }
        for s in snapshots
    ]
