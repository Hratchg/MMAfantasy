"""Query functions for matchup data.

Provides read-only query functions to load features, striking accuracy,
and style matchup counts for the matchup comparison system.
All functions accept a SQLAlchemy Session and return plain Python data
structures. Uses SQLAlchemy 2.0 select() style (D-16).
"""

from __future__ import annotations

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session, aliased

from ufc_prediction.models.computed_feature import ComputedFeature
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.round_stats import RoundStats


def get_latest_features(session: Session, fighter_id: int) -> dict[str, object] | None:
    """Get the most recent computed features dict for a fighter.

    Args:
        session: SQLAlchemy session.
        fighter_id: Fighter's database ID.

    Returns:
        Features dict from the most recent ComputedFeature row,
        or None if no features exist for this fighter.
    """
    stmt = (
        select(ComputedFeature.features)
        .where(ComputedFeature.fighter_id == fighter_id)
        .order_by(ComputedFeature.as_of_date.desc())
        .limit(1)
    )
    return session.scalar(stmt)


def get_striking_accuracy(session: Session, fighter_id: int) -> float | None:
    """Compute striking accuracy from career round stats aggregation.

    Resolves RESEARCH Pitfall 1: striking accuracy is not in the features
    JSON, so we compute it from round_stats (sig_str_landed / sig_str_attempted).

    Args:
        session: SQLAlchemy session.
        fighter_id: Fighter's database ID.

    Returns:
        Career striking accuracy as a float (0.0-1.0), or None if
        no round stats exist or zero attempts.
    """
    stmt = select(
        func.sum(RoundStats.sig_strikes_landed).label("total_landed"),
        func.sum(RoundStats.sig_strikes_attempted).label("total_attempted"),
    ).where(RoundStats.fighter_id == fighter_id)

    row = session.execute(stmt).first()
    if row is None or row.total_attempted is None or row.total_attempted == 0:
        return None

    result: float = row.total_landed / row.total_attempted
    return result


def get_style_matchup_counts(
    session: Session,
) -> dict[tuple[str, str], dict[str, int]]:
    """Aggregate historical style-vs-style win counts from computed features.

    Joins fights with computed features for both fighters (at the time of
    the fight) to extract style tags. Excludes draws/no-contests where
    winner_id IS NULL (D-08). Normalizes style pairs alphabetically to
    avoid double-counting.

    Returns:
        Dict mapping (style_a, style_b) tuples (alphabetically sorted) to
        {"a_wins": int, "b_wins": int, "total": int}.
    """
    cf_a = aliased(ComputedFeature, name="cf_a")
    cf_b = aliased(ComputedFeature, name="cf_b")

    # Extract style tags from JSON features column
    style_a_col = cf_a.features["style_tag"].as_string()
    style_b_col = cf_b.features["style_tag"].as_string()

    # Normalize: always put alphabetically-first style as "style_lo"
    style_lo = case(
        (style_a_col <= style_b_col, style_a_col),
        else_=style_b_col,
    )
    style_hi = case(
        (style_a_col <= style_b_col, style_b_col),
        else_=style_a_col,
    )

    # Who won: did the "lo" style's fighter win?
    # We need to track which fighter_id corresponds to style_lo
    lo_fighter_id = case(
        (style_a_col <= style_b_col, Fight.fighter_a_id),
        else_=Fight.fighter_b_id,
    )

    lo_won = case(
        (Fight.winner_id == lo_fighter_id, 1),
        else_=0,
    )

    stmt = (
        select(
            style_lo.label("style_lo"),
            style_hi.label("style_hi"),
            func.count().label("total"),
            func.sum(lo_won).label("lo_wins"),
        )
        .select_from(Fight)
        .join(
            cf_a,
            (cf_a.fight_id == Fight.id) & (cf_a.fighter_id == Fight.fighter_a_id),
        )
        .join(
            cf_b,
            (cf_b.fight_id == Fight.id) & (cf_b.fighter_id == Fight.fighter_b_id),
        )
        .where(Fight.winner_id != None)  # noqa: E711 -- Exclude draws/NC
        .where(style_a_col != style_b_col)  # Only cross-style matchups
        .group_by(style_lo, style_hi)
    )

    rows = session.execute(stmt).all()

    result: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        total = row.total
        lo_wins = row.lo_wins or 0
        hi_wins = total - lo_wins
        result[(row.style_lo, row.style_hi)] = {
            "a_wins": lo_wins,
            "b_wins": hi_wins,
            "total": total,
        }

    return result
