"""Fighter ratings endpoint (API-01)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ufc_prediction.api.deps import get_db
from ufc_prediction.api.routers._lookup import resolve_fighter_or_candidates
from ufc_prediction.api.schemas import (
    DivisionRating,
    FighterRatingResponse,
    FighterSearchResponse,
)
from ufc_prediction.elo.fighter_queries import (
    get_fighter_detail,
    get_fighter_divisions,
    get_fighter_domain_elo,
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
    fighter = resolve_fighter_or_candidates(db, name)
    if isinstance(fighter, FighterSearchResponse):
        return fighter

    # Single match -- build full rating response
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
