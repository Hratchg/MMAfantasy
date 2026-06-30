"""Pydantic validation schemas for Kaggle CSV row data.

These models serve as the boundary validation layer between untrusted CSV data
and the application's database models. Per D-06, malformed rows are skipped and
logged rather than crashing the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from pydantic import BaseModel, field_validator

# All 14 valid UFC weight classes
VALID_WEIGHT_CLASSES: frozenset[str] = frozenset(
    {
        "Flyweight",
        "Bantamweight",
        "Featherweight",
        "Lightweight",
        "Welterweight",
        "Middleweight",
        "Light Heavyweight",
        "Heavyweight",
        "Women's Strawweight",
        "Women's Flyweight",
        "Women's Bantamweight",
        "Women's Featherweight",
        "Catch Weight",
        "Open Weight",
    }
)


class FightStats(BaseModel):
    """Per-fighter aggregated stats for a single fight (maps to RoundStats with round_number=0)."""

    sig_strikes_landed: int | None = None
    sig_strikes_attempted: int | None = None
    takedowns_landed: int | None = None
    takedowns_attempted: int | None = None
    submission_attempts: int | None = None
    reversals: int | None = None
    control_time_seconds: int | None = None
    knockdowns: int | None = None
    head_strikes_landed: int | None = None
    body_strikes_landed: int | None = None
    leg_strikes_landed: int | None = None
    distance_strikes_landed: int | None = None
    clinch_strikes_landed: int | None = None
    ground_strikes_landed: int | None = None


class FightRow(BaseModel):
    """Validated fight data ready for database insertion."""

    event_name: str
    event_date: date
    location: str | None = None
    fighter_a_name: str
    fighter_b_name: str
    winner_name: str | None = None  # None for draws/NC
    weight_class: str
    method: str | None = None
    method_detail: str | None = None
    round_finished: int | None = None
    time_finished: str | None = None
    is_title_fight: bool = False
    num_rounds: int = 3
    fighter_a_stats: FightStats | None = None
    fighter_b_stats: FightStats | None = None

    @field_validator("weight_class")
    @classmethod
    def validate_weight_class(cls, v: str) -> str:
        """Validate weight class against the canonical set."""
        if v not in VALID_WEIGHT_CLASSES:
            msg = f"Unknown weight class: '{v}'. Valid: {sorted(VALID_WEIGHT_CLASSES)}"
            raise ValueError(msg)
        return v

    @field_validator("event_date")
    @classmethod
    def validate_not_future(cls, v: date) -> date:
        """Per D-07: lightweight temporal check -- date must not be in the future.

        Allows 1 day of slack for timezone differences.
        """
        tomorrow = date.today() + timedelta(days=1)
        if v > tomorrow:
            msg = f"Event date {v} is in the future (today is {date.today()})"
            raise ValueError(msg)
        return v


class FighterRow(BaseModel):
    """Validated fighter profile data."""

    name: str
    height_inches: float | None = None
    reach_inches: float | None = None
    leg_reach_inches: float | None = None  # Not available in Kaggle datasets
    stance: str | None = None
    date_of_birth: date | None = None


@dataclass
class IngestResult:
    """Tracks ingestion statistics. Per D-06: skip and log on validation failures."""

    accepted: int = 0
    rejected: int = 0
    updated: int = 0
    rejections: list[dict[str, object]] = field(default_factory=list)

    def log_rejection(self, row_index: int, raw_row: dict[str, object], error: str) -> None:
        """Record a rejected row with truncated data for debugging."""
        self.rejected += 1
        self.rejections.append(
            {
                "row": row_index,
                "data": {
                    k: str(v)[:100]
                    for k, v in raw_row.items()
                    if k in ("R_fighter", "B_fighter", "date")
                },
                "error": str(error)[:500],
            }
        )

    def summary(self) -> str:
        """Return a human-readable summary of the ingestion result."""
        return (
            f"Ingestion complete: {self.accepted} accepted, "
            f"{self.updated} updated, {self.rejected} rejected"
        )
