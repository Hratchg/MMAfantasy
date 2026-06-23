from datetime import date, datetime
from typing import Any

from sqlalchemy import JSON, Date, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column

from ufc_prediction.db.base import Base


class ComputedFeature(Base):
    __tablename__ = "computed_features"

    id: Mapped[int] = mapped_column(primary_key=True)
    fighter_id: Mapped[int] = mapped_column(ForeignKey("fighters.id"), nullable=False)
    fight_id: Mapped[int] = mapped_column(ForeignKey("fights.id"), nullable=False)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    feature_set_version: Mapped[str] = mapped_column(String(50), nullable=False)
    features: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=func.now())
