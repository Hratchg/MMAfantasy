"""Matchup comparison orchestrator.

Assembles physical differentials, Elo comparison, striking/grappling
matrices, and style analysis into a single MatchupResult (D-13, D-15).
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from ufc_prediction.elo.fighter_queries import (
    get_fighter_detail,
    get_fighter_divisions,
    get_fighter_domain_elo,
)
from ufc_prediction.matchup.matrix import (
    GrapplingMatrix,
    StrikingMatrix,
    compute_grappling_matrix,
    compute_striking_matrix,
)
from ufc_prediction.matchup.physical import (
    PhysicalDifferentials,
    compute_physical_differentials,
)
from ufc_prediction.matchup.queries import (
    get_latest_features,
    get_striking_accuracy,
    get_style_matchup_counts,
)
from ufc_prediction.matchup.style_analysis import (
    StyleMatchupWinRate,
    compute_style_index,
    compute_style_vs_style_win_rates,
)


@dataclass
class EloComparison:
    """Elo comparison between two fighters across overall and domain ratings."""

    fighter_a_overall: float | None
    fighter_b_overall: float | None
    fighter_a_striking: float | None
    fighter_b_striking: float | None
    fighter_a_grappling: float | None
    fighter_b_grappling: float | None
    fighter_a_division: str | None
    fighter_b_division: str | None


@dataclass
class StyleAnalysis:
    """Style analysis for a matchup: tags, indices, and historical win rates."""

    fighter_a_style: str | None
    fighter_b_style: str | None
    fighter_a_index: float | None
    fighter_b_index: float | None
    matchup_win_rates: list[StyleMatchupWinRate]
    cross_division_warning: str | None


@dataclass
class MatchupResult:
    """Full matchup comparison result assembling all analysis sections."""

    fighter_a_name: str
    fighter_b_name: str
    physical: PhysicalDifferentials
    elo: EloComparison
    striking: StrikingMatrix
    grappling: GrapplingMatrix
    style: StyleAnalysis


def _find_common_division(
    divisions_a: list[str],
    divisions_b: list[str],
) -> tuple[str | None, str | None, str | None]:
    """Find common division or pick most recent for each fighter.

    Returns:
        (division_a, division_b, cross_division_warning).
        If a common division exists, both will be the same and warning is None.
        If no common division, each gets their first division and a warning is set.
    """
    common = set(divisions_a) & set(divisions_b)
    if common:
        # Pick alphabetically first common division for determinism
        div = sorted(common)[0]
        return div, div, None

    div_a = divisions_a[0] if divisions_a else None
    div_b = divisions_b[0] if divisions_b else None

    warning = None
    if div_a and div_b and div_a != div_b:
        warning = (
            f"Cross-division comparison: Fighter A competes in {div_a}, "
            f"Fighter B competes in {div_b}. Elo ratings may not be directly comparable."
        )
    elif not div_a or not div_b:
        warning = "One or both fighters have no recorded divisions."

    return div_a, div_b, warning


def build_matchup(
    session: Session,
    fighter_a: object,
    fighter_b: object,
) -> MatchupResult:
    """Build a full matchup comparison between two fighters.

    Orchestrates all sub-modules to assemble physical differentials,
    Elo comparison, striking/grappling matrices, and style analysis.

    Args:
        session: SQLAlchemy session for database queries.
        fighter_a: Fighter object (must have id, name, physical attributes).
        fighter_b: Fighter object (must have id, name, physical attributes).

    Returns:
        MatchupResult with all sections populated. Missing data is
        represented as None values throughout.
    """
    # 1. Get divisions and find common ground
    divisions_a = get_fighter_divisions(session, fighter_a.id)
    divisions_b = get_fighter_divisions(session, fighter_b.id)
    div_a, div_b, cross_warning = _find_common_division(divisions_a, divisions_b)

    # 2. Physical differentials
    physical = compute_physical_differentials(fighter_a, fighter_b)

    # 3. Overall Elo
    detail_a = get_fighter_detail(session, fighter_a.id, div_a)
    detail_b = get_fighter_detail(session, fighter_b.id, div_b)

    # 4. Domain Elo
    domain_a = (
        get_fighter_domain_elo(session, fighter_a.id, div_a)
        if div_a
        else {"striking_elo": None, "grappling_elo": None}
    )
    domain_b = (
        get_fighter_domain_elo(session, fighter_b.id, div_b)
        if div_b
        else {"striking_elo": None, "grappling_elo": None}
    )

    elo = EloComparison(
        fighter_a_overall=detail_a["elo"],
        fighter_b_overall=detail_b["elo"],
        fighter_a_striking=domain_a["striking_elo"],
        fighter_b_striking=domain_b["striking_elo"],
        fighter_a_grappling=domain_a["grappling_elo"],
        fighter_b_grappling=domain_b["grappling_elo"],
        fighter_a_division=div_a or detail_a.get("division"),
        fighter_b_division=div_b or detail_b.get("division"),
    )

    # 5. Latest features
    features_a = get_latest_features(session, fighter_a.id)
    features_b = get_latest_features(session, fighter_b.id)

    # 6. Striking accuracy from round stats
    accuracy_a = get_striking_accuracy(session, fighter_a.id)
    accuracy_b = get_striking_accuracy(session, fighter_b.id)

    # 7. Striking matrix
    striking = compute_striking_matrix(features_a, features_b, accuracy_a, accuracy_b)

    # 8. Grappling matrix
    grappling = compute_grappling_matrix(features_a, features_b)

    # 9. Style tags from features
    style_a = features_a.get("style_tag") if features_a else None
    style_b = features_b.get("style_tag") if features_b else None

    # 10. Style index from domain Elo
    index_a = compute_style_index(domain_a["striking_elo"], domain_a["grappling_elo"])
    index_b = compute_style_index(domain_b["striking_elo"], domain_b["grappling_elo"])

    # 11. Style matchup win rates
    matchup_counts = get_style_matchup_counts(session)
    win_rates = compute_style_vs_style_win_rates(matchup_counts)

    style = StyleAnalysis(
        fighter_a_style=style_a,
        fighter_b_style=style_b,
        fighter_a_index=index_a,
        fighter_b_index=index_b,
        matchup_win_rates=win_rates,
        cross_division_warning=cross_warning,
    )

    # 12. Assemble and return
    return MatchupResult(
        fighter_a_name=fighter_a.name,
        fighter_b_name=fighter_b.name,
        physical=physical,
        elo=elo,
        striking=striking,
        grappling=grappling,
        style=style,
    )
