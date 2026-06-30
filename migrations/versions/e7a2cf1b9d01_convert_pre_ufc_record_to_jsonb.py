"""convert pre_ufc_record from JSON to JSONB

Revision ID: e7a2cf1b9d01
Revises: b394e0f2770e
Create Date: 2026-04-23 03:15:00.000000

Postgres `json` does not support equality, which breaks `SELECT DISTINCT` on
any query pulling the fighters row (e.g. `search_fighters`). JSONB supports
equality, is better indexed, and is the idiomatic choice for storing
structured JSON in Postgres.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e7a2cf1b9d01"
down_revision: Union[str, None] = "b394e0f2770e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "fighters",
        "pre_ufc_record",
        type_=sa.dialects.postgresql.JSONB(),
        existing_type=sa.JSON(),
        existing_nullable=True,
        postgresql_using="pre_ufc_record::jsonb",
    )


def downgrade() -> None:
    op.alter_column(
        "fighters",
        "pre_ufc_record",
        type_=sa.JSON(),
        existing_type=sa.dialects.postgresql.JSONB(),
        existing_nullable=True,
        postgresql_using="pre_ufc_record::json",
    )
