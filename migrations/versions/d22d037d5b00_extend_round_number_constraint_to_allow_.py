"""extend round_number constraint to allow zero

Revision ID: d22d037d5b00
Revises: 96ab1f67fb42
Create Date: 2026-04-14 18:25:27.055956

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d22d037d5b00"
down_revision: Union[str, Sequence[str], None] = "96ab1f67fb42"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Extend round_number constraint from 1-5 to 0-5 to allow fight-level aggregates."""
    op.drop_constraint(op.f("ck_round_stats_valid_round_number"), "round_stats", type_="check")
    op.create_check_constraint(
        op.f("ck_round_stats_valid_round_number"),
        "round_stats",
        "round_number BETWEEN 0 AND 5",
    )


def downgrade() -> None:
    """Restore round_number constraint to 1-5."""
    op.drop_constraint(op.f("ck_round_stats_valid_round_number"), "round_stats", type_="check")
    op.create_check_constraint(
        op.f("ck_round_stats_valid_round_number"),
        "round_stats",
        "round_number BETWEEN 1 AND 5",
    )
