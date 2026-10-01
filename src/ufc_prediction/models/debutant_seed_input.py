"""DebutantSeedInput ORM model — Sherdog pre-UFC record behind each debutant Elo seed.

One row per seeded fighter, mirroring the locked 14-column
``data/sherdog/pre_ufc_records.csv`` schema (Plan 43-01). This table is the
single source of truth for debutant Elo seeds: ``ufc elo compute`` and the
serve path (``ml.inference_features``) derive seeds from it with
``elo.seed.derive_seed``; the CSV only feeds ``ufc db backfill-pre-ufc-seeds``
and remains an export of ``scripts/ingest_pre_ufc_records_v25.py``.

Why not ``fighters.pre_ufc_record``: that JSONB (Phase 22 ``ufc scrape
sherdog``) covers a wider population (every matched fighter, any source), has
no ``org_tier``, and feeds the frozen model's ``pre_ufc_win_pct_diff`` feature,
so seeding from it would change who is seeded and repurposing it would touch
the feature substrate.

Columns:
- Seed inputs (NOT NULL): ``n_pre_ufc_fights``, ``win_rate`` (float8, so a
  Python float round-trips bit-exactly), ``org_tier``.
- Provenance (NOT NULL): ``sherdog_url``, ``scraped_at`` (timestamptz).
- Descriptive counts / ``last_organization`` (NULL allowed): carried for the
  CSV export and audits, not read by the seed formula.

CHECK constraints mirror what ``derive_seed`` accepts so a bad row cannot land
in the table the seeds are read from.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from ufc_prediction.db.base import Base

# Locked tier taxonomy (elo.seed._TIER_BONUSES keys; scripts/ingest_pre_ufc_records_v25.py).
ORG_TIERS: tuple[str, ...] = ("major", "regional", "local", "none")


class DebutantSeedInput(Base):
    """Sherdog pre-UFC record for one seeded debutant (debutant Elo seed input)."""

    __tablename__ = "debutant_seed_inputs"

    fighter_id: Mapped[int] = mapped_column(
        ForeignKey("fighters.id", ondelete="CASCADE"), primary_key=True
    )
    sherdog_url: Mapped[str] = mapped_column(String(500), nullable=False)
    n_pre_ufc_fights: Mapped[int] = mapped_column(Integer, nullable=False)
    wins: Mapped[int | None] = mapped_column(Integer)
    losses: Mapped[int | None] = mapped_column(Integer)
    draws: Mapped[int | None] = mapped_column(Integer)
    nc_dq: Mapped[int | None] = mapped_column(Integer)
    win_rate: Mapped[float] = mapped_column(Float, nullable=False)
    kos: Mapped[int | None] = mapped_column(Integer)
    submissions: Mapped[int | None] = mapped_column(Integer)
    decisions: Mapped[int | None] = mapped_column(Integer)
    last_organization: Mapped[str | None] = mapped_column(String(200))
    org_tier: Mapped[str] = mapped_column(String(16), nullable=False)
    scraped_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        CheckConstraint(
            "org_tier IN (" + ", ".join(f"'{t}'" for t in ORG_TIERS) + ")",
            name="org_tier",
        ),
        CheckConstraint("win_rate >= 0 AND win_rate <= 1", name="win_rate"),
        CheckConstraint("n_pre_ufc_fights >= 0", name="n_pre_ufc_fights"),
    )
