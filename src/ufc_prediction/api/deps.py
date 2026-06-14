"""Database session dependency for FastAPI endpoints."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy.orm import Session

from ufc_prediction.db.session import SessionLocal


def get_db() -> Generator[Session, None, None]:
    """Yield a database session, closing it when the request completes."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
