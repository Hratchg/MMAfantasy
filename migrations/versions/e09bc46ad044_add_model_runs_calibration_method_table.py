"""add model_runs calibration_method table

Revision ID: e09bc46ad044
Revises: 59981c08e056
Create Date: 2026-05-16 10:53:18.901218

Creates the ``model_runs`` table per Phase 22 D-09 (forward-looking schema for
Phase 26 CALIB-V22-01). Phase 22 ships SCHEMA ONLY; Phase 26 writes rows via
``ml/persistence.py`` extension (the existing JSON-sidecar reader path stays
UNCHANGED in Phase 22 per D-09).

Schema per RESEARCH Finding 12 type tightening:
- Bounded ``String(N)`` for ``model_name`` / ``model_version`` /
  ``calibration_method`` / ``model_artifact_sha256`` (SHA-256 hex == 64 chars).
- ``sa.Date`` for ``cutoff_date`` (matches Event.date precedent).
- JSONB for ``metadata_json`` (matches the e7a2cf1b9d01 + fighter.pre_ufc_record
  precedent).
- Composite Index ``(model_name, model_version)`` — "find runs of model+version".
- Single-col Index ``(trained_at)`` — "recent runs ordered by training date".

All constraints use ``op.f()`` per ``db/base.py`` naming convention dict:
``pk_model_runs``. Indexes named explicitly per RESEARCH Finding 4 +
PATTERNS §model_runs ("ix_model_runs_model_name_model_version",
"ix_model_runs_trained_at").

Banned imports per Pitfall #1 / Finding 11: nothing under ``ufc_prediction.ml.*``.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = "e09bc46ad044"
down_revision: Union[str, Sequence[str], None] = "59981c08e056"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema — create model_runs table + 2 indexes."""
    op.create_table(
        "model_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("model_version", sa.String(length=50), nullable=False),
        sa.Column("calibration_method", sa.String(length=20), nullable=True),
        sa.Column("cutoff_date", sa.Date(), nullable=False),
        sa.Column("n_training_fights", sa.Integer(), nullable=True),
        sa.Column("n_test_fights", sa.Integer(), nullable=True),
        sa.Column(
            "trained_at",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("model_artifact_sha256", sa.String(length=64), nullable=True),
        sa.Column("metadata_json", JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_runs")),
    )
    op.create_index(
        "ix_model_runs_model_name_model_version",
        "model_runs",
        ["model_name", "model_version"],
    )
    op.create_index(
        "ix_model_runs_trained_at",
        "model_runs",
        ["trained_at"],
    )


def downgrade() -> None:
    """Downgrade schema — drop indexes then table (reverse of upgrade)."""
    op.drop_index("ix_model_runs_trained_at", table_name="model_runs")
    op.drop_index(
        "ix_model_runs_model_name_model_version",
        table_name="model_runs",
    )
    op.drop_table("model_runs")
