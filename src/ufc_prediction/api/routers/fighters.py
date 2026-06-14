"""Fighter ratings endpoint (API-01)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ufc_prediction.api.deps import get_db
from ufc_prediction.api.schemas import (
    DivisionRating,
    FighterRatingResponse,
    FighterSearchResponse,
    FighterSearchResult,
)
from ufc_prediction.elo.fighter_queries import (
    get_fighter_detail,
    get_fighter_divisions,
    get_fighter_domain_elo,
    search_fighters,
)

router = APIRouter(tags=["fighters"])


@router.get("/fighters/{name}")
def get_fighter(
    name: str, db: Session = Depends(get_db)
) -> FighterSearchResponse | FighterRatingResponse:
    """Get fighter ratings by name.

    Returns full rating details for a single match, or a candidate list
    when multiple fighters match the search term (D-03, D-13).
    """
    matches = search_fighters(db, name)

    if not matches:
        raise HTTPException(status_code=404, detail=f"Fighter '{name}' not found")

    if len(matches) > 1:
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

    # Single match -- build full rating response
    fighter = matches[0]
    divisions = get_fighter_divisions(db, fighter.id)
    division_ratings = []

    for div in divisions:
        detail = get_fighter_detail(db, fighter.id, div)
        domain = get_fighter_domain_elo(db, fighter.id, div)
        division_ratings.append(
            DivisionRating(
                division=div,
                elo=detail.get("elo"),
                striking_elo=domain.get("striking_elo"),
                grappling_elo=domain.get("grappling_elo"),
                wins=detail.get("wins", 0),
                losses=detail.get("losses", 0),
                total_fights=detail.get("total_fights", 0),
                last_fight_date=detail.get("last_fight_date"),
            )
        )

    return FighterRatingResponse(
        name=fighter.name,
        id=fighter.id,
        divisions=division_ratings,
    )
