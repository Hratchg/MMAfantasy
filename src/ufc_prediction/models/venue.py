"""Venue ORM model — Phase 22 TRAVEL-V22-01.

Per CONTEXT D-08 + REVISION-03: 10-column schema with ``geocode_source`` audit
column. Populated by one-off ``scripts/backfill_venue_geocodes.py`` from existing
``events.location`` strings via Nominatim (NOT a runtime dep per D-07).

Schema rationale (CONTEXT D-08 + REVISION-03; RESEARCH Findings 4, 9, 12):
- 10 persisted columns + ``created_at``; CSV header is the 10-col persisted subset.
- ``name`` (String(300), NOT NULL): canonical venue name (e.g. "T-Mobile Arena").
- ``city`` (String(200), NULL), ``state`` (String(100), NULL): nullable per D-08.
- ``country`` (String(100), NOT NULL): always populated by Nominatim addressdetails.
- ``lat``/``lon`` (Float, NOT NULL): required by the data-integrity guard in
  ``tests/data/test_venues_csv.py`` (no null geocoords; manual-edit if Nominatim missed).
- ``timezone_iana`` (String(64), NOT NULL): IANA tz e.g. "America/Los_Angeles";
  derived from (lat, lon) via TimezoneFinder() (NOT TimezoneFinderL — coastline
  accuracy matters for venues like UFC Fight Island / T-Mobile Arena per RESEARCH F6).
- ``n_events`` (Integer, NULL): count of events resolving to this venue (sort key
  for operator review by importance; D-08).
- ``geocode_source`` (String(100), NULL): audit-trail provenance per REVISION-03;
  one of ``nominatim:YYYY-MM-DD`` / ``manual-edit:YYYY-MM-DD`` /
  ``unknown:imported-from-events.location``.

Banned imports per Pitfall #1 / Finding 11: nothing under ``ufc_prediction.ml.*``.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from ufc_prediction.db.base import Base


class Venue(Base):
    """Venue ORM model — Phase 22 TRAVEL-V22-01."""

    __tablename__ = "venues"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    city: Mapped[str | None] = mapped_column(String(200))
    state: Mapped[str | None] = mapped_column(String(100))
    country: Mapped[str] = mapped_column(String(100), nullable=False)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lon: Mapped[float] = mapped_column(Float, nullable=False)
    timezone_iana: Mapped[str] = mapped_column(String(64), nullable=False)
    n_events: Mapped[int | None] = mapped_column(Integer)
    geocode_source: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(default=func.now())
