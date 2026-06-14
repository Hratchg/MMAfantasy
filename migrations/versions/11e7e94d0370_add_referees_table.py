"""add referees table

Revision ID: 11e7e94d0370
Revises: 4c9cb5ced391
Create Date: 2026-05-16 00:10:34.593161

Creates the ``referees`` table per Phase 22 REF-V22-01.

Schema rationale (CONTEXT D-01, D-10; RESEARCH Findings 2, 4):
- ``name`` (String(200), NOT NULL) — raw display name as scraped from UFCStats.
- ``normalized_name`` (String(200), NOT NULL, UNIQUE) — canonical lowercase-hyphen
  form via scraper/referee_normalize.normalize_referee_name. Used as the dedup key
  by upsert_referee().
- ``alias_list`` (JSONB, NULL) — raw-name variants encountered in HTML
  (e.g., ["Herb Dean", "Herbert Dean"]); appended on duplicate-detection in upsert.
- ``created_at`` (DateTime, server_default=now()) — matches fight_odds.scraped_at style.
- ``events.referee_id`` (Integer, NULL, FK -> referees.id) — per-event ref attribution
  per CONTEXT D-10 lossy-aggregation note (per-fight placement deferred to v2.3+).
  NULLABLE per RESEARCH Pitfall #3: adding NOT NULL FK to populated 5,799-row table
  fails IntegrityError. Backfill is operator-driven post-Phase-22 (NOT in scope).

All constraints use op.f() per db/base.py naming convention dict:
pk_referees, uq_referees_normalized_name, fk_events_referee_id_referees.

Banned imports (Pitfall #1 / Finding 11): nothing under ``ufc_prediction.ml.*``.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = '11e7e94d0370'
down_revision: Union[str, Sequence[str], None] = '4c9cb5ced391'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema — create referees table + add events.referee_id FK."""
    op.create_table(
        "referees",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("normalized_name", sa.String(length=200), nullable=False),
        sa.Column("alias_list", JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_referees")),
        sa.UniqueConstraint(
            "normalized_name",
            name=op.f("uq_referees_normalized_name"),
        ),
    )

    op.add_column(
        "events",
        sa.Column("referee_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        op.f("fk_events_referee_id_referees"),
        "events",
        "referees",
        ["referee_id"],
        ["id"],
    )


def downgrade() -> None:
    """Downgrade schema — drop FK then column then table (reverse order)."""
    op.drop_constraint(
        op.f("fk_events_referee_id_referees"),
        "events",
        type_="foreignkey",
    )
    op.drop_column("events", "referee_id")
    op.drop_table("referees")
