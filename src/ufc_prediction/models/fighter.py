from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import Date, Float, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ufc_prediction.db.base import Base


class Fighter(Base):
    __tablename__ = "fighters"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    nickname: Mapped[str | None] = mapped_column(String(200))
    height_inches: Mapped[float | None] = mapped_column(Float)
    reach_inches: Mapped[float | None] = mapped_column(Float)
    leg_reach_inches: Mapped[float | None] = mapped_column(Float)
    stance: Mapped[str | None] = mapped_column(String(20))
    date_of_birth: Mapped[date | None] = mapped_column(Date)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    source_id: Mapped[str | None] = mapped_column(String(200))
    sherdog_url: Mapped[str | None] = mapped_column(String(500))
    pre_ufc_record: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    updated_at: Mapped[datetime] = mapped_column(default=func.now(), onupdate=func.now())

    aliases: Mapped[list[FighterAlias]] = relationship(back_populates="fighter")


class FighterAlias(Base):
    __tablename__ = "fighter_aliases"

    id: Mapped[int] = mapped_column(primary_key=True)
    fighter_id: Mapped[int] = mapped_column(ForeignKey("fighters.id"), nullable=False)
    alias_name: Mapped[str] = mapped_column(String(200), nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False)

    fighter: Mapped[Fighter] = relationship(back_populates="aliases")

    __table_args__ = (
        UniqueConstraint("alias_name", "source", name="uq_fighter_aliases_name_source"),
    )
