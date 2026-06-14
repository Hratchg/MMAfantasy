"""Division rankings endpoint (API-02)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ufc_prediction.api.deps import get_db
from ufc_prediction.api.schemas import RankedFighter, RankingsResponse
from ufc_prediction.elo.fighter_queries import (
    get_division_rankings,
    list_divisions,
    resolve_weight_class,
)

router = APIRouter(tags=["rankings"])


@router.get("/rankings/{division}", response_model=RankingsResponse)
def get_rankings(
    division: str,
    top: int = Query(default=15, ge=1, le=100),
    db: Session = Depends(get_db),
) -> RankingsResponse:
    """Get top fighters in a division ranked by Elo.

    The division parameter supports partial matching (e.g., 'light' matches
    'Lightweight'). The top parameter caps at 100 (T-09-02).
    """
    matches = resolve_weight_class(division)

    if not matches:
        valid = ", ".join(list_divisions())
        raise HTTPException(
            status_code=404,
            detail=f"Division '{division}' not found. Valid: {valid}",
        )

    if len(matches) > 1:
        raise HTTPException(
            status_code=400,
            detail=f"Ambiguous division '{division}'. Matches: {', '.join(matches)}",
        )

    resolved = matches[0]
    rows = get_division_rankings(db, resolved, limit=top)

    fighters = [
        RankedFighter(
            rank=idx,
            name=row["name"],
            fighter_id=row["fighter_id"],
            elo=row["elo"],
            fights=row["fights"],
            last_active=row["last_date"],
        )
        for idx, row in enumerate(rows, start=1)
    ]

    return RankingsResponse(division=resolved, fighters=fighters)
