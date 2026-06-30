"""add fight_odds table

Revision ID: 4c9cb5ced391
Revises: e7a2cf1b9d01
Create Date: 2026-04-24 11:09:47.891113

Creates the ``fight_odds`` table per Phase 13 Plan 02.

Schema rationale:
- Dedicated table (not columns on ``fights``) per RESEARCH anti-pattern:
  a dozen+ odds columns bolted onto ``fights`` would be unwieldy.
- Composite PK (fight_id, fighter_id) enforces idempotent upsert.
- CASCADE FKs to fights.id and fighters.id — orphaned rows are disallowed
  by DB rather than left for app code (T-13-02-01).
- Raw American moneyline columns (opening_ml, closing_range_min_ml,
  closing_range_max_ml) preserve D-03's raw-data requirement.
- Computed implied probability columns (opening_implied_prob,
  closing_implied_prob) are vig-removed via D-02 proportional normalization.
- Missing odds (pre-2007, D-07) stored as SQL NULL — never zero.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4c9cb5ced391"
down_revision: Union[str, Sequence[str], None] = "e7a2cf1b9d01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "fight_odds",
        sa.Column("fight_id", sa.Integer(), nullable=False),
        sa.Column("fighter_id", sa.Integer(), nullable=False),
        sa.Column("opening_ml", sa.Integer(), nullable=True),
        sa.Column("closing_range_min_ml", sa.Integer(), nullable=True),
        sa.Column("closing_range_max_ml", sa.Integer(), nullable=True),
        sa.Column("opening_implied_prob", sa.Float(), nullable=True),
        sa.Column("closing_implied_prob", sa.Float(), nullable=True),
        sa.Column(
            "source",
            sa.String(length=50),
            nullable=False,
            server_default="bestfightodds",
        ),
        sa.Column(
            "scraped_at",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["fight_id"],
            ["fights.id"],
            name=op.f("fk_fight_odds_fight_id_fights"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["fighter_id"],
            ["fighters.id"],
            name=op.f("fk_fight_odds_fighter_id_fighters"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("fight_id", "fighter_id", name="pk_fight_odds"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("fight_odds")
