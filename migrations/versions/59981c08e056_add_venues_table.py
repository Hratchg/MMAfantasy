"""add venues table

Revision ID: 59981c08e056
Revises: 11e7e94d0370
Create Date: 2026-05-16 10:49:30.154539

Creates the ``venues`` table per Phase 22 TRAVEL-V22-01.

Schema rationale (CONTEXT D-08 + REVISION-03; RESEARCH Findings 4, 9, 12):
- 10-column schema matching ``data/venues.csv`` header exactly:
  ``venue_id, name, city, state, country, lat, lon, timezone_iana, n_events,
  geocode_source`` (CSV header) -> persisted as ``id, name, city, state,
  country, lat, lon, timezone_iana, n_events, geocode_source`` (+ server-side
  ``created_at`` audit column).
- ``geocode_source`` (REVISION-03) audit-trail column for ``nominatim:YYYY-MM-DD``
  / ``manual-edit:YYYY-MM-DD`` / ``unknown:imported-from-events.location``
  provenance.
- ``events.venue_id`` (Integer, NULL, FK -> ``venues.id``) — NULLABLE per
  Pitfall #3 (adding NOT NULL FK to populated 5,799-row ``events`` table fails
  IntegrityError). Backfill is deferred to v2.3+ Backlog (re-run scripts/
  backfill_venue_geocodes.py after Nominatim cool-off; or hand-fill top-N).
- bulk_insert via lowercase ``sa.table()`` / ``sa.column()`` per Pitfall #2
  (NOT ORM ``Venue.__table__`` — insulates migration from future ORM drift).

Data delivery (Plan 22-03 Task 3 PARTIAL operator-PROCEED verdict):
- ``data/venues.csv`` ships 52 rows (~30% of 173 distinct location strings;
  ~10% by event count — Las Vegas, 611 events, in the missing 121).
- Schema is forward-compatible: future operator can INSERT additional rows
  into the populated ``venues`` table without re-running this migration.

All constraints use ``op.f()`` per ``db/base.py`` naming convention dict:
``pk_venues``, ``fk_events_venue_id_venues``.

Banned imports per Pitfall #1 / Finding 11: nothing under ``ufc_prediction.ml.*``.
"""

import csv
from pathlib import Path
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "59981c08e056"
down_revision: Union[str, Sequence[str], None] = "11e7e94d0370"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema — create venues, bulk-insert from data/venues.csv, add events.venue_id FK."""
    op.create_table(
        "venues",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=300), nullable=False),
        sa.Column("city", sa.String(length=200), nullable=True),
        sa.Column("state", sa.String(length=100), nullable=True),
        sa.Column("country", sa.String(length=100), nullable=False),
        sa.Column("lat", sa.Float(), nullable=False),
        sa.Column("lon", sa.Float(), nullable=False),
        sa.Column("timezone_iana", sa.String(length=64), nullable=False),
        sa.Column("n_events", sa.Integer(), nullable=True),
        sa.Column("geocode_source", sa.String(length=100), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_venues")),
    )

    # Pitfall #2: lowercase sa.table()/sa.column() ad-hoc constructs (NOT ORM
    # Venue.__table__) — insulates migration from future ORM drift.
    venues_table = sa.table(
        "venues",
        sa.column("id", sa.Integer),
        sa.column("name", sa.String),
        sa.column("city", sa.String),
        sa.column("state", sa.String),
        sa.column("country", sa.String),
        sa.column("lat", sa.Float),
        sa.column("lon", sa.Float),
        sa.column("timezone_iana", sa.String),
        sa.column("n_events", sa.Integer),
        sa.column("geocode_source", sa.String),
    )

    # migrations/versions/<rev>.py -> repo root is parents[2]
    csv_path = Path(__file__).resolve().parents[2] / "data" / "venues.csv"
    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "id": int(r["venue_id"]),
                    "name": r["name"],
                    "city": r["city"] or None,
                    "state": r["state"] or None,
                    "country": r["country"],
                    "lat": float(r["lat"]),
                    "lon": float(r["lon"]),
                    "timezone_iana": r["timezone_iana"],
                    "n_events": int(r["n_events"]) if r["n_events"] else None,
                    "geocode_source": r["geocode_source"] or None,
                }
            )
    if rows:
        op.bulk_insert(venues_table, rows)

    op.add_column(
        "events",
        sa.Column("venue_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        op.f("fk_events_venue_id_venues"),
        "events",
        "venues",
        ["venue_id"],
        ["id"],
    )


def downgrade() -> None:
    """Downgrade schema — drop FK, column, then table (reverse of upgrade)."""
    op.drop_constraint(
        op.f("fk_events_venue_id_venues"),
        "events",
        type_="foreignkey",
    )
    op.drop_column("events", "venue_id")
    op.drop_table("venues")
