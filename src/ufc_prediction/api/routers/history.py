"""Elo history endpoint (API-04)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ufc_prediction.api.deps import get_db
from ufc_prediction.api.schemas import (
    EloHistoryPoint,
    EloHistoryResponse,
    FighterSearchResponse,
    FighterSearchResult,
)
from ufc_prediction.elo.fighter_queries import (
    get_elo_history,
    get_fighter_detail,
    search_fighters,
)

router = APIRouter(tags=["history"])


@router.get("/fighters/{name}/history")
def get_fighter_history(
    name: str,
    division: str | None = Query(default=None),
    elo_type: str = Query(default="overall"),
    db: Session = Depends(get_db),
) -> FighterSearchResponse | EloHistoryResponse:
    """Get a fighter's chronological Elo trajectory.

    Optionally filter by division and elo_type (overall, striking, grappling).
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

    fighter = matches[0]
    history_data = get_elo_history(db, fighter.id, division=division, elo_type=elo_type)

    history = [
        EloHistoryPoint(
            fight_date=h["fight_date"],
            division=h["division"],
            elo_before=h["elo_before"],
            elo_after=h["elo_after"],
            elo_after_shrinkage=h["elo_after_shrinkage"],
        )
        for h in history_data
    ]

    return EloHistoryResponse(
        name=fighter.name,
        fighter_id=fighter.id,
        history=history,
    )
