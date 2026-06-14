"""relax num_rounds constraint to 1-5

Revision ID: 30a14a290754
Revises: c1d121894bba
Create Date: 2026-04-15 01:27:27.322674

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '30a14a290754'
down_revision: Union[str, Sequence[str], None] = 'c1d121894bba'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_constraint("valid_num_rounds", "fights", type_="check")
    op.create_check_constraint(
        "valid_num_rounds", "fights", sa.text("num_rounds BETWEEN 1 AND 5")
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("valid_num_rounds", "fights", type_="check")
    op.create_check_constraint(
        "valid_num_rounds", "fights", sa.text("num_rounds IN (3, 5)")
    )
