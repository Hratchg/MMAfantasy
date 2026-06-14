"""Pydantic models for Sherdog parsed data.

These models represent fighter profiles and fight records scraped from
Sherdog.com, and the computed pre-UFC career summary stats stored on
the Fighter model as a JSON blob.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel


class SherdogFight(BaseModel):
    """A single fight from a Sherdog fighter profile."""

    result: str  # "win", "loss", "draw", "nc"
    opponent_name: str
    method: str  # "KO/TKO", "Submission", "Decision", etc.
    event_name: str
    event_date: date | None = None  # Parsed from date string
    round_finished: int | None = None


class SherdogFighterProfile(BaseModel):
    """Parsed from a Sherdog fighter page."""

    name: str
    sherdog_url: str
    fights: list[SherdogFight]


class PreUFCRecord(BaseModel):
    """Computed pre-UFC summary stats (per D-10).

    Stored as JSON on Fighter.pre_ufc_record column.
    """

    total_wins: int
    total_losses: int
    total_draws: int
    total_fights: int
    win_pct: float  # 0.0 to 1.0
    ko_finish_rate: float  # KO wins / total wins
    sub_finish_rate: float  # Sub wins / total wins
    decision_rate: float  # Decision wins / total wins
    career_years: float  # Years between first and last pre-UFC fight
    fights: list[SherdogFight]  # The raw fight list for reference
