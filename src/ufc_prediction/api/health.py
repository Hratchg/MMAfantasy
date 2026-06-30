"""Health + readiness endpoints (API-V24-04, Phase 35).

`/health` is liveness — returns 200 + version, never touches the DB.
`/ready` runs `SELECT 1` with a 1-second timeout to verify DB connectivity;
returns 503 on failure or timeout. Both routes are mounted at the root
(no `/api/v1` prefix) and are auth-exempt + rate-limit-exempt.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from ufc_prediction.db.session import SessionLocal

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

_READY_TIMEOUT_SECONDS: float = 1.0


def _run_select_one() -> int | None:
    """Synchronous SELECT 1 — runs in threadpool via asyncio.to_thread."""
    session = SessionLocal()
    try:
        return session.execute(text("SELECT 1")).scalar()
    finally:
        session.close()


@router.get("/health")
async def liveness(request: Request) -> dict[str, str]:
    """Liveness probe — no DB touch, returns app version."""
    return {"status": "ok", "version": request.app.version}


@router.get("/ready")
async def readiness() -> JSONResponse:
    """Readiness probe — 200 if SELECT 1 succeeds within 1s, else 503."""
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_run_select_one),
            timeout=_READY_TIMEOUT_SECONDS,
        )
        # Tolerate any non-None scalar; we only care that the round-trip closed.
        if result is None:
            return JSONResponse(
                status_code=503,
                content={"status": "db_unavailable", "error": "no_result"},
            )
        return JSONResponse(status_code=200, content={"status": "ready"})
    except TimeoutError:
        return JSONResponse(
            status_code=503,
            content={"status": "db_unavailable", "error": "timeout"},
        )
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "db_unavailable", "error": str(exc)},
        )
