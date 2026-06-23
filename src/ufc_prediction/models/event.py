from datetime import date, datetime

from sqlalchemy import Date, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column

from ufc_prediction.db.base import Base


class Event(Base):
    """UFC events in this table follow the ufcstats-canonical naming and ID scheme.

    The training and inference pipelines filter to ``source == 'ufcstats'`` rows
    via ``ml/queries.py:138`` and treat all other source rows (Kaggle historical,
    etc.) as audit-only context. Cross-reference: ``ml/queries.py:load_computed_features()``
    enforces temporal validity via ``as_of_date <= event_date`` (TEMPORAL-V24-01).
    """

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    location: Mapped[str | None] = mapped_column(String(300))
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    source_url: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    # Phase 22 REF-V22-01 — NULLABLE per Pitfall #3 (existing 5,799-row events
    # table cannot tolerate NOT NULL FK addition without backfill).
    referee_id: Mapped[int | None] = mapped_column(ForeignKey("referees.id"))
    # Phase 22 TRAVEL-V22-01 — NULLABLE per Pitfall #3 (same constraint as
    # referee_id; venues backfill is operator-driven per Plan 22-03 Task 3
    # PARTIAL operator-PROCEED verdict and v2.3+ Backlog).
    venue_id: Mapped[int | None] = mapped_column(ForeignKey("venues.id"))
