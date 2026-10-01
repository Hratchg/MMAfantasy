"""add debutant_seed_inputs table (Sherdog debutant Elo seeds in the DB)

Revision ID: 9fc47fdd75b6
Revises: e09bc46ad044
Create Date: 2026-09-29 12:00:00.000000

Operator decision D3 option C (2026-09-29): the Sherdog debutant-seed inputs
move from the untracked ``data/sherdog/pre_ufc_records.csv`` into the DB so the
Docker serving image (which ships no ``data/``) serves seeded debutants. The
table mirrors the CSV's locked 14-column schema, one row per seeded fighter.

- Seed inputs NOT NULL: ``n_pre_ufc_fights``, ``win_rate`` (float8 — a Python
  float round-trips bit-exactly, so the derived seeds equal ``load_seeds(csv)``),
  ``org_tier``.
- Provenance NOT NULL: ``sherdog_url``, ``scraped_at`` (timestamptz).
- CHECKs mirror ``elo.seed.derive_seed``'s domain (locked tier set, win rate in
  [0, 1], non-negative fight count).
- ``fighter_id`` is the PK and an FK to ``fighters`` (ON DELETE CASCADE, the
  ``fight_odds`` precedent).

Schema only: rows are loaded by ``ufc db backfill-pre-ufc-seeds --csv …`` and,
going forward, by ``scripts/ingest_pre_ufc_records_v25.py``.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "9fc47fdd75b6"
down_revision: Union[str, Sequence[str], None] = "e09bc46ad044"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "debutant_seed_inputs",
        sa.Column("fighter_id", sa.Integer(), nullable=False),
        sa.Column("sherdog_url", sa.String(length=500), nullable=False),
        sa.Column("n_pre_ufc_fights", sa.Integer(), nullable=False),
        sa.Column("wins", sa.Integer(), nullable=True),
        sa.Column("losses", sa.Integer(), nullable=True),
        sa.Column("draws", sa.Integer(), nullable=True),
        sa.Column("nc_dq", sa.Integer(), nullable=True),
        sa.Column("win_rate", sa.Float(), nullable=False),
        sa.Column("kos", sa.Integer(), nullable=True),
        sa.Column("submissions", sa.Integer(), nullable=True),
        sa.Column("decisions", sa.Integer(), nullable=True),
        sa.Column("last_organization", sa.String(length=200), nullable=True),
        sa.Column("org_tier", sa.String(length=16), nullable=False),
        sa.Column("scraped_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "org_tier IN ('major', 'regional', 'local', 'none')",
            name=op.f("ck_debutant_seed_inputs_org_tier"),
        ),
        sa.CheckConstraint(
            "win_rate >= 0 AND win_rate <= 1",
            name=op.f("ck_debutant_seed_inputs_win_rate"),
        ),
        sa.CheckConstraint(
            "n_pre_ufc_fights >= 0",
            name=op.f("ck_debutant_seed_inputs_n_pre_ufc_fights"),
        ),
        sa.ForeignKeyConstraint(
            ["fighter_id"],
            ["fighters.id"],
            name=op.f("fk_debutant_seed_inputs_fighter_id_fighters"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("fighter_id", name=op.f("pk_debutant_seed_inputs")),
    )


def downgrade() -> None:
    op.drop_table("debutant_seed_inputs")
