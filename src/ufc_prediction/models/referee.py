"""Referee ORM model — Phase 22 REF-V22-01.

Per CONTEXT D-10 (REVISION-02): consolidates the existing Pydantic-only
referee extraction at scraper/parse_fight_detail.py:80-90 into a persisted
DB row. ``normalized_name`` is the canonical key (lowercase-hyphen via
scraper/referee_normalize.normalize_referee_name); ``alias_list`` records
raw-name variations seen in scraped HTML (e.g., ``["Herb Dean", "Herbert Dean"]``).

Banned imports per Pitfall #1 / Finding 11: nothing under ``ufc_prediction.ml.*``.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ufc_prediction.db.base import Base


class Referee(Base):
    """Referee ORM model — Phase 22 REF-V22-01.

    Per CONTEXT D-10 (REVISION-02): consolidates the existing Pydantic-only
    referee extraction at scraper/parse_fight_detail.py:80-90 into a persisted
    DB row. ``normalized_name`` is the canonical key (lowercase-hyphen via
    scraper/referee_normalize.normalize_referee_name); ``alias_list`` records
    raw-name variations seen in scraped HTML (e.g., ``["Herb Dean", "Herbert Dean"]``).
    """

    __tablename__ = "referees"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(200), nullable=False)
    alias_list: Mapped[list[str] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(default=func.now())

    __table_args__ = (
        UniqueConstraint("normalized_name", name="uq_referees_normalized_name"),
    )
