from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Date, Float, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ufc_prediction.db.base import Base

if TYPE_CHECKING:
    from ufc_prediction.models.fight import Fight
    from ufc_prediction.models.fighter import Fighter


class EloSnapshot(Base):
    __tablename__ = "elo_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    fighter_id: Mapped[int] = mapped_column(ForeignKey("fighters.id"), nullable=False)
    fight_id: Mapped[int] = mapped_column(ForeignKey("fights.id"), nullable=False)
    division: Mapped[str] = mapped_column(String(50), nullable=False)
    elo_type: Mapped[str] = mapped_column(String(20), nullable=False)
    elo_before: Mapped[float] = mapped_column(Float, nullable=False)
    elo_after: Mapped[float] = mapped_column(Float, nullable=False)
    elo_after_shrinkage: Mapped[float] = mapped_column(Float, nullable=False)
    k_factor_used: Mapped[float] = mapped_column(Float, nullable=False)
    fight_date: Mapped[date] = mapped_column(Date, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=func.now())

    fighter: Mapped[Fighter] = relationship()
    fight: Mapped[Fight] = relationship(back_populates="elo_snapshots")
