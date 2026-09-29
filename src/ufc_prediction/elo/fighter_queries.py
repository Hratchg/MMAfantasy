"""Query functions for fighter lookup and division rankings.

Provides read-only query functions used by the CLI fighter commands.
All functions accept a SQLAlchemy Session and return plain Python data
structures (no Rich/Typer dependencies). Uses SQLAlchemy 2.0 select() style.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import date, timedelta
from typing import Any

from sqlalchemy import case, func, or_, select
from sqlalchemy.orm import Session

from ufc_prediction.dedup.source_priority import SOURCE_PRIORITY, prefer_canonical
from ufc_prediction.elo.asof import RatedFight, pre_fight_rating
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
_ALL_DIVISIONS: frozenset[str] = frozenset(
    {
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
    }
)

# Rankable divisions exclude Catch Weight and Open Weight.
# Sorted longest-first for greedy substring matching.
_RANKABLE_DIVISIONS: list[str] = sorted(
    [wc for wc in _ALL_DIVISIONS if wc not in {"Catch Weight", "Open Weight"}],
    key=lambda x: -len(x),
)


_LIKE_ESCAPE = "\\"

# Upper bound on the candidate list the API returns for an ambiguous name.
MAX_FIGHTER_CANDIDATES = 25


def _escape_like(value: str) -> str:
    """Escape LIKE metacharacters so user input matches literally."""
    return (
        value.replace(_LIKE_ESCAPE, _LIKE_ESCAPE * 2)
        .replace("%", _LIKE_ESCAPE + "%")
        .replace("_", _LIKE_ESCAPE + "_")
    )


def _source_rank() -> Any:
    """SQL expression ranking Fighter.source by SOURCE_PRIORITY (unknown last)."""
    return case(SOURCE_PRIORITY, value=Fighter.source, else_=99)


def _collapse_by_name(fighters: Sequence[Fighter]) -> list[Fighter]:
    """Collapse same-name rows (one real person across ingest sources) to the
    canonical row, preserving first-seen name order."""
    by_name: dict[str, list[Fighter]] = {}
    for f in fighters:
        by_name.setdefault(f.name.lower(), []).append(f)
    return [
        prefer_canonical(group, source_key=lambda f: f.source, tiebreak_key=lambda f: f.id)
        for group in by_name.values()
    ]


def search_fighters(session: Session, name: str, *, limit: int | None = None) -> list[Fighter]:
    """Search fighters by name or alias (case-insensitive ILIKE).

    Uses parameterized queries via SQLAlchemy bind parameters (T-04-01).
    ``%``, ``_`` and ``\\`` in ``name`` match literally. Rows are ordered by
    name, then source priority, then id, so a ``limit`` never keeps a
    lower-priority duplicate while dropping its canonical twin.
    """
    pattern = f"%{_escape_like(name)}%"
    alias_match = (
        select(FighterAlias.id)
        .where(FighterAlias.fighter_id == Fighter.id)
        .where(FighterAlias.alias_name.ilike(pattern, escape=_LIKE_ESCAPE))
        .exists()
    )
    stmt = (
        select(Fighter)
        .where(or_(Fighter.name.ilike(pattern, escape=_LIKE_ESCAPE), alias_match))
        .order_by(func.lower(Fighter.name), _source_rank(), Fighter.id)
    )
    if limit is not None:
        stmt = stmt.limit(limit)
    return list(session.scalars(stmt).all())


def resolve_fighter_candidates(
    session: Session,
    name: str,
    *,
    limit: int = MAX_FIGHTER_CANDIDATES,
) -> list[Fighter]:
    """Resolve a user-supplied name to canonical fighter candidates.

    Cross-source duplicates (the same person ingested from ufcstats and a
    Kaggle dataset) collapse to one canonical row via ``prefer_canonical``.
    An exact case-insensitive name match wins outright over substring hits,
    so a full name always resolves to a single fighter. Otherwise returns
    at most ``limit`` distinct candidates.
    """
    needle = name.strip()
    if not needle:
        return []

    exact_stmt = (
        select(Fighter)
        .where(func.lower(Fighter.name) == needle.lower())
        .order_by(_source_rank(), Fighter.id)
    )
    exact = list(session.scalars(exact_stmt).all())
    if exact:
        return _collapse_by_name(exact)

    # Each real person has at most one row per source, so over-fetching by
    # the number of known sources keeps `limit` distinct names in range.
    rows = search_fighters(session, needle, limit=limit * len(SOURCE_PRIORITY))
    return _collapse_by_name(rows)[:limit]


def get_latest_overall_elo_bulk(
    session: Session,
    fighter_ids: Sequence[int],
) -> dict[int, tuple[float, str]]:
    """Latest overall ``(elo_after_shrinkage, division)`` per fighter, in one query.

    Fighters with no overall snapshot are absent from the result.
    """
    if not fighter_ids:
        return {}
    ranked = (
        select(
            EloSnapshot.fighter_id,
            EloSnapshot.elo_after_shrinkage,
            EloSnapshot.division,
            func.row_number()
            .over(
                partition_by=EloSnapshot.fighter_id,
                order_by=(EloSnapshot.fight_date.desc(), EloSnapshot.id.desc()),
            )
            .label("rn"),
        )
        .where(EloSnapshot.elo_type == "overall")
        .where(EloSnapshot.fighter_id.in_(list(fighter_ids)))
        .subquery()
    )
    stmt = select(ranked.c.fighter_id, ranked.c.elo_after_shrinkage, ranked.c.division).where(
        ranked.c.rn == 1
    )
    return {
        row.fighter_id: (float(row.elo_after_shrinkage), row.division)
        for row in session.execute(stmt).all()
    }


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
        .where(Fight.winner_id.is_not(None))
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
    *,
    as_of: date | None = None,
) -> list[dict[str, Any]]:
    """Get top fighters in a division ranked by current Elo.

    Uses only the most recent snapshot per fighter to avoid duplicates, then
    applies the engine's inactivity regression up to ``as_of`` (default
    today) via ``elo.asof.pre_fight_rating``, so long-inactive fighters are
    ranked on the rating they would carry into their next fight rather than
    their last post-fight value. ``elo`` in the result is that regressed,
    shrunk rating; ``last_date`` is still the fighter's last fight in the
    division.
    """
    as_of = as_of or date.today()
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
    # priority before truncating to `limit`. Every fighter in the division is
    # fetched (no over-fetch window): a Kaggle twin can out-rate its ufcstats
    # twin by hundreds of places, and any fixed window would let the Kaggle
    # row through as canonical. A division is at most a few thousand rows.
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

    current = _current_division_ratings(
        session, [r.fighter_id for r in canonical_rows], division, as_of
    )
    ranked = sorted(
        ((current.get(r.fighter_id, float(r.elo_after_shrinkage)), r) for r in canonical_rows),
        key=lambda pair: pair[0],
        reverse=True,
    )[:limit]

    return [
        {
            "name": row.name,
            "elo": elo,
            "fighter_id": row.fighter_id,
            "fights": row.fight_count,
            "last_date": row.fight_date,
        }
        for elo, row in ranked
    ]


def _current_division_ratings(
    session: Session,
    fighter_ids: Sequence[int],
    division: str,
    as_of: date,
) -> dict[int, float]:
    """Each fighter's ``division`` rating carried forward to ``as_of``.

    Replays the fighter's full overall history through
    ``pre_fight_rating`` so inactivity regression matches the engine (every
    gap past the threshold regresses all division ratings; the final gap is
    measured from the last fight in any division). The replay is fed
    ``elo_after_shrinkage``: shrinkage, regression and transfer are all linear
    pulls toward the initial rating, so this yields the regressed display
    rating directly.
    """
    if not fighter_ids:
        return {}
    stmt = (
        select(
            EloSnapshot.fighter_id,
            EloSnapshot.fight_date,
            EloSnapshot.division,
            EloSnapshot.elo_after_shrinkage,
        )
        .where(EloSnapshot.elo_type == "overall")
        .where(EloSnapshot.fighter_id.in_(list(fighter_ids)))
        .order_by(EloSnapshot.fighter_id, EloSnapshot.fight_date, EloSnapshot.id)
    )
    history: dict[int, list[RatedFight]] = defaultdict(list)
    for row in session.execute(stmt).all():
        history[row.fighter_id].append(
            RatedFight(
                fight_date=row.fight_date,
                division=row.division,
                elo_after=float(row.elo_after_shrinkage),
            )
        )

    result: dict[int, float] = {}
    for fighter_id, fights in history.items():
        # pre_fight_rating needs history strictly before its as_of; a fight
        # dated on (or after) as_of means zero inactivity, so step past it.
        effective = max(as_of, fights[-1].fight_date + timedelta(days=1))
        result[fighter_id] = pre_fight_rating(fights, as_of=effective, division=division)
    return result


def resolve_weight_class(input_str: str) -> list[str]:
    """Resolve partial weight class input to matching divisions.

    Excludes Catch Weight and Open Weight from results. An exact
    case-insensitive division name resolves to that division alone (so
    'Heavyweight' is not ambiguous with 'Light Heavyweight'); otherwise
    falls back to case-insensitive substring matching.
    """
    needle = input_str.strip().lower()
    exact = [wc for wc in _RANKABLE_DIVISIONS if needle == wc.lower()]
    if exact:
        return exact
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
        results.append(
            {
                "name": row.name,
                "fighter_id": row.id,
                "division": row.division,
                "elo": row.elo,
                "striking_elo": domain["striking_elo"],
                "grappling_elo": domain["grappling_elo"],
                "last_fight_date": str(row.last_fight_date) if row.last_fight_date else "",
            }
        )
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
