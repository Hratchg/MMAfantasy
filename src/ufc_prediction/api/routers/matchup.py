"""Matchup comparison endpoint (API-03)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ufc_prediction.api.deps import get_db
from ufc_prediction.api.schemas import (
    EloComparisonModel,
    FighterSearchResponse,
    FighterSearchResult,
    GrapplingModel,
    MatchupResponse,
    PhysicalModel,
    StrikingModel,
    StyleMatchupWinRateModel,
    StyleModel,
)
from ufc_prediction.elo.fighter_queries import get_fighter_detail, search_fighters
from ufc_prediction.matchup.compare import build_matchup
from ufc_prediction.models.fighter import Fighter

router = APIRouter(tags=["matchup"])


def _resolve_single_fighter(db: Session, name: str) -> Fighter | FighterSearchResponse | None:
    """Resolve a name to a single fighter, or return None / search response."""
    matches = search_fighters(db, name)
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    # Multiple matches -- return search response for caller to handle
    results = []
    for fighter in matches:
        detail = get_fighter_detail(db, fighter.id)
        results.append(
            FighterSearchResult(
                name=fighter.name,
                id=fighter.id,
                division=detail.get("division"),
                elo=detail.get("elo"),
            )
        )
    return FighterSearchResponse(count=len(results), results=results)


@router.get("/matchup")
def get_matchup(
    a: str = Query(..., description="Fighter A name"),
    b: str = Query(..., description="Fighter B name"),
    db: Session = Depends(get_db),
) -> MatchupResponse | FighterSearchResponse:
    """Compare two fighters across all matchup dimensions.

    Returns physical differentials, Elo comparison, striking/grappling
    matrices, and style analysis.
    """
    # Resolve fighter A
    result_a = _resolve_single_fighter(db, a)
    if result_a is None:
        raise HTTPException(status_code=404, detail=f"Fighter '{a}' not found")
    if isinstance(result_a, FighterSearchResponse):
        return result_a

    # Resolve fighter B
    result_b = _resolve_single_fighter(db, b)
    if result_b is None:
        raise HTTPException(status_code=404, detail=f"Fighter '{b}' not found")
    if isinstance(result_b, FighterSearchResponse):
        return result_b

    # Same fighter check
    if result_a.id == result_b.id:
        raise HTTPException(
            status_code=400,
            detail="Cannot compare a fighter to themselves",
        )

    matchup_result = build_matchup(db, result_a, result_b)

    # Convert dataclasses to Pydantic models
    physical = PhysicalModel(
        height_diff=matchup_result.physical.height_diff,
        reach_diff=matchup_result.physical.reach_diff,
        leg_reach_diff=matchup_result.physical.leg_reach_diff,
        age_diff=matchup_result.physical.age_diff,
        fighter_a_age=matchup_result.physical.fighter_a_age,
        fighter_b_age=matchup_result.physical.fighter_b_age,
    )

    elo = EloComparisonModel(
        fighter_a_overall=matchup_result.elo.fighter_a_overall,
        fighter_b_overall=matchup_result.elo.fighter_b_overall,
        fighter_a_striking=matchup_result.elo.fighter_a_striking,
        fighter_b_striking=matchup_result.elo.fighter_b_striking,
        fighter_a_grappling=matchup_result.elo.fighter_a_grappling,
        fighter_b_grappling=matchup_result.elo.fighter_b_grappling,
        fighter_a_division=matchup_result.elo.fighter_a_division,
        fighter_b_division=matchup_result.elo.fighter_b_division,
    )

    striking = StrikingModel(
        a_sig_str_per_min=matchup_result.striking.a_sig_str_per_min,
        a_striking_accuracy=matchup_result.striking.a_striking_accuracy,
        a_strike_defense=matchup_result.striking.a_strike_defense,
        b_sig_str_per_min=matchup_result.striking.b_sig_str_per_min,
        b_striking_accuracy=matchup_result.striking.b_striking_accuracy,
        b_strike_defense=matchup_result.striking.b_strike_defense,
    )

    grappling = GrapplingModel(
        a_td_rate=matchup_result.grappling.a_td_rate,
        a_td_accuracy=matchup_result.grappling.a_td_accuracy,
        a_td_defense=matchup_result.grappling.a_td_defense,
        b_td_rate=matchup_result.grappling.b_td_rate,
        b_td_accuracy=matchup_result.grappling.b_td_accuracy,
        b_td_defense=matchup_result.grappling.b_td_defense,
        a_ctrl_time=matchup_result.grappling.a_ctrl_time,
        b_ctrl_time=matchup_result.grappling.b_ctrl_time,
    )

    win_rates = [
        StyleMatchupWinRateModel(
            style_a=wr.style_a,
            style_b=wr.style_b,
            style_a_wins=wr.style_a_wins,
            style_b_wins=wr.style_b_wins,
            total=wr.total,
            style_a_win_pct=wr.style_a_win_pct,
            sufficient_data=wr.sufficient_data,
        )
        for wr in matchup_result.style.matchup_win_rates
    ]

    style = StyleModel(
        fighter_a_style=matchup_result.style.fighter_a_style,
        fighter_b_style=matchup_result.style.fighter_b_style,
        fighter_a_index=matchup_result.style.fighter_a_index,
        fighter_b_index=matchup_result.style.fighter_b_index,
        matchup_win_rates=win_rates,
        cross_division_warning=matchup_result.style.cross_division_warning,
    )

    return MatchupResponse(
        fighter_a_name=matchup_result.fighter_a_name,
        fighter_b_name=matchup_result.fighter_b_name,
        physical=physical,
        elo=elo,
        striking=striking,
        grappling=grappling,
        style=style,
    )
