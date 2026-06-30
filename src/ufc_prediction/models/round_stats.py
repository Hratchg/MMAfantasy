from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, ForeignKey, Integer, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ufc_prediction.db.base import Base

if TYPE_CHECKING:
    from ufc_prediction.models.fight import Fight
    from ufc_prediction.models.fighter import Fighter


class RoundStats(Base):
    __tablename__ = "round_stats"

    id: Mapped[int] = mapped_column(primary_key=True)
    fight_id: Mapped[int] = mapped_column(ForeignKey("fights.id"), nullable=False)
    fighter_id: Mapped[int] = mapped_column(ForeignKey("fighters.id"), nullable=False)
    round_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # Striking
    sig_strikes_landed: Mapped[int | None] = mapped_column(Integer)
    sig_strikes_attempted: Mapped[int | None] = mapped_column(Integer)
    head_strikes_landed: Mapped[int | None] = mapped_column(Integer)
    body_strikes_landed: Mapped[int | None] = mapped_column(Integer)
    leg_strikes_landed: Mapped[int | None] = mapped_column(Integer)
    distance_strikes_landed: Mapped[int | None] = mapped_column(Integer)
    clinch_strikes_landed: Mapped[int | None] = mapped_column(Integer)
    ground_strikes_landed: Mapped[int | None] = mapped_column(Integer)

    # Grappling
    takedowns_landed: Mapped[int | None] = mapped_column(Integer)
    takedowns_attempted: Mapped[int | None] = mapped_column(Integer)
    submission_attempts: Mapped[int | None] = mapped_column(Integer)
    reversals: Mapped[int | None] = mapped_column(Integer)
    control_time_seconds: Mapped[int | None] = mapped_column(Integer)
    knockdowns: Mapped[int | None] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(default=func.now())

    fight: Mapped[Fight] = relationship(back_populates="rounds")
    fighter: Mapped[Fighter] = relationship()

    __table_args__ = (CheckConstraint("round_number BETWEEN 0 AND 5", name="valid_round_number"),)
