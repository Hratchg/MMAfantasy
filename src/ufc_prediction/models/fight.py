from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, CheckConstraint, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ufc_prediction.db.base import Base

if TYPE_CHECKING:
    from ufc_prediction.models.elo_snapshot import EloSnapshot
    from ufc_prediction.models.event import Event
    from ufc_prediction.models.fighter import Fighter
    from ufc_prediction.models.round_stats import RoundStats


class Fight(Base):
    __tablename__ = "fights"

    id: Mapped[int] = mapped_column(primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"), nullable=False)
    fighter_a_id: Mapped[int] = mapped_column(ForeignKey("fighters.id"), nullable=False)
    fighter_b_id: Mapped[int] = mapped_column(ForeignKey("fighters.id"), nullable=False)
    winner_id: Mapped[int | None] = mapped_column(ForeignKey("fighters.id"))
    weight_class: Mapped[str] = mapped_column(String(50), nullable=False)
    method: Mapped[str | None] = mapped_column(String(50))
    method_detail: Mapped[str | None] = mapped_column(String(100))
    round_finished: Mapped[int | None] = mapped_column(Integer)
    time_finished: Mapped[str | None] = mapped_column(String(10))
    is_title_fight: Mapped[bool] = mapped_column(Boolean, default=False)
    num_rounds: Mapped[int] = mapped_column(Integer, default=3)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    source_url: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(default=func.now())

    event: Mapped[Event] = relationship()
    fighter_a: Mapped[Fighter] = relationship(foreign_keys=[fighter_a_id])
    fighter_b: Mapped[Fighter] = relationship(foreign_keys=[fighter_b_id])
    winner: Mapped[Fighter | None] = relationship(foreign_keys=[winner_id])
    rounds: Mapped[list[RoundStats]] = relationship(back_populates="fight")
    elo_snapshots: Mapped[list[EloSnapshot]] = relationship(back_populates="fight")

    __table_args__ = (
        CheckConstraint("num_rounds BETWEEN 1 AND 5", name="valid_num_rounds"),
        CheckConstraint("fighter_a_id != fighter_b_id", name="different_fighters"),
    )
