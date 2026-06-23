"""ModelRun ORM model — Phase 22 forward-looking schema for Phase 26 CALIB-V22-01.

Per CONTEXT D-09 + RESEARCH Finding 12: schema tightening replaces TEXT/INTEGER
with bounded types; JSONB ``metadata_json`` matches the e7a2cf1b9d01 +
fighter.pre_ufc_record precedent; composite + single-col indexes provision
the "find runs of model+version" and "recent runs" lookups.

Phase 22 ships SCHEMA ONLY; Phase 26 writes rows via ml/persistence.py extension
— the existing JSON-sidecar reader path stays UNCHANGED in Phase 22 (D-09).

Schema rationale (CONTEXT D-09; RESEARCH Finding 12):
- ``model_name`` (String(100), NOT NULL): e.g. "xgb_v2", "xgb_v2_meta_v22".
- ``model_version`` (String(50), NOT NULL): semver-ish version tag.
- ``calibration_method`` (String(20), NULL): 'sigmoid' | 'isotonic' | NULL (uncalibrated).
- ``cutoff_date`` (Date, NOT NULL): train/test boundary; matches Event.date precedent.
- ``n_training_fights`` / ``n_test_fights`` (Integer, NULL): cardinality at training time.
- ``trained_at`` (DateTime, NOT NULL, server_default=now()): UTC timestamp.
- ``model_artifact_sha256`` (String(64), NULL): SHA-256 hex (exactly 64 chars) of the .joblib.
- ``metadata_json`` (JSONB, NULL): full sidecar contents for forward-compat.

Indexes:
- ``ix_model_runs_model_name_model_version`` composite — supports "find runs of model+version".
- ``ix_model_runs_trained_at`` single-col — supports "recent runs ordered by training date".

Banned imports per Pitfall #1 / Finding 11: nothing under ``ufc_prediction.ml.*``.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import Date, DateTime, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ufc_prediction.db.base import Base


class ModelRun(Base):
    """ModelRun ORM model — Phase 22 D-09 forward-looking schema for Phase 26."""

    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    calibration_method: Mapped[str | None] = mapped_column(String(20))  # 'sigmoid'|'isotonic'|NULL
    cutoff_date: Mapped[date] = mapped_column(Date, nullable=False)
    n_training_fights: Mapped[int | None] = mapped_column(Integer)
    n_test_fights: Mapped[int | None] = mapped_column(Integer)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False,
    )
    model_artifact_sha256: Mapped[str | None] = mapped_column(String(64))  # SHA-256 hex
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    __table_args__ = (
        Index("ix_model_runs_model_name_model_version", "model_name", "model_version"),
        Index("ix_model_runs_trained_at", "trained_at"),
    )
