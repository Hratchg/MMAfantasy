"""FightOdds SQLAlchemy model.

Stores betting odds per (fight, fighter) linked from BestFightOdds via
the ufcscraper package. Separate table (not columns on fights) per
RESEARCH anti-pattern and D-03 (raw ML + computed implied prob both
stored).

Threat mitigations:
- T-13-02-01: FK constraints prevent orphaned rows (ondelete="CASCADE")
- T-13-02-02: Composite PK (fight_id, fighter_id) enforces idempotent upsert
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    PrimaryKeyConstraint,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from ufc_prediction.db.base import Base


class FightOdds(Base):
    """Per-(fight, fighter) betting odds row from BestFightOdds."""

    __tablename__ = "fight_odds"

    fight_id: Mapped[int] = mapped_column(
        ForeignKey("fights.id", ondelete="CASCADE"), nullable=False
    )
    fighter_id: Mapped[int] = mapped_column(
        ForeignKey("fighters.id", ondelete="CASCADE"), nullable=False
    )

    # Raw American moneyline values (D-03: preserve raw data).
    opening_ml: Mapped[int | None] = mapped_column(Integer, nullable=True)
    closing_range_min_ml: Mapped[int | None] = mapped_column(Integer, nullable=True)
    closing_range_max_ml: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Computed vig-removed implied probabilities (D-02 per-fight normalization).
    opening_implied_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    closing_implied_prob: Mapped[float | None] = mapped_column(Float, nullable=True)

    source: Mapped[str] = mapped_column(String(50), nullable=False, server_default="bestfightodds")
    scraped_at: Mapped[datetime] = mapped_column(
        DateTime(), nullable=False, server_default=func.now()
    )

    __table_args__ = (PrimaryKeyConstraint("fight_id", "fighter_id", name="pk_fight_odds"),)
