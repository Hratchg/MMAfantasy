"""Shared fighter-name resolution for the fighter, history and matchup routes."""

from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.orm import Session

from ufc_prediction.api.schemas import FighterSearchResponse, FighterSearchResult
from ufc_prediction.elo.fighter_queries import (
    get_latest_overall_elo_bulk,
    resolve_fighter_candidates,
)
from ufc_prediction.models.fighter import Fighter


def resolve_fighter_or_candidates(db: Session, name: str) -> Fighter | FighterSearchResponse:
    """Resolve ``name`` to one canonical fighter, or a capped candidate list.

    Cross-source duplicates collapse to the canonical row and an exact name
    wins over substring hits (``resolve_fighter_candidates``). Candidate Elo
    is fetched in a single query. Raises 404 when nothing matches.
    """
    matches = resolve_fighter_candidates(db, name)
    if not matches:
        raise HTTPException(status_code=404, detail=f"Fighter '{name}' not found")
    if len(matches) == 1:
        return matches[0]

    ratings = get_latest_overall_elo_bulk(db, [f.id for f in matches])
    results = []
    for fighter in matches:
        rating = ratings.get(fighter.id)
        results.append(
            FighterSearchResult(
                name=fighter.name,
                id=fighter.id,
                division=rating[1] if rating else None,
                elo=rating[0] if rating else None,
            )
        )
    return FighterSearchResponse(count=len(results), results=results)
