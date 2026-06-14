"""Scraper-specific Pydantic models for parsed HTML data.

These intermediate models represent the bridge between raw HTML from UFCStats.com
and the existing DB schemas (FightRow, FighterRow, FightStats) in data/schemas.py.
All string fields use raw strings from HTML. Conversion to typed values happens
in the ingest orchestrator using existing parsers from data/parsers.py.
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class EventSummary(BaseModel):
    """Parsed from event listing page. One per event row."""

    name: str
    date_str: str  # Raw "April 11, 2026"
    location: str
    url: str
    is_upcoming: bool = False


class FightSummary(BaseModel):
    """Parsed from event detail page. One per fight row."""

    fight_url: str
    fighter_a_name: str
    fighter_a_url: str
    fighter_b_name: str
    fighter_b_url: str
    winner: str | None  # "fighter_a", "fighter_b", None for draw/NC
    outcome: str  # "win", "draw", "nc"
    weight_class_raw: str  # Raw text before parsing
    is_title_fight: bool
    method: str | None
    method_detail: str | None
    round_finished: int | None
    time_finished: str | None


class EventDetail(BaseModel):
    """Parsed from event detail page. Complete event with fights."""

    name: str
    date_str: str
    location: str
    url: str
    fights: list[FightSummary]

    @field_validator("fights")
    @classmethod
    def check_cardinality(cls, v: list[FightSummary]) -> list[FightSummary]:
        if len(v) < 1:
            msg = f"Event detail cardinality: expected >=1 fights, got {len(v)}"
            raise ValueError(msg)
        return v


class RoundStatsRaw(BaseModel):
    """Parsed stats for one fighter in one round (or totals)."""

    fighter_name: str
    knockdowns: str  # Raw string, e.g., "1"
    sig_str: str  # "X of Y" format
    sig_str_pct: str  # "65%"
    total_str: str  # "X of Y"
    td: str  # "X of Y"
    td_pct: str  # "50%"
    sub_att: str  # "0"
    rev: str  # "0"
    ctrl: str  # "M:SS"


class SigStrikesRaw(BaseModel):
    """Parsed significant strikes breakdown for one fighter."""

    fighter_name: str
    sig_str: str  # "X of Y"
    sig_str_pct: str
    head: str  # "X of Y"
    body: str  # "X of Y"
    leg: str  # "X of Y"
    distance: str  # "X of Y"
    clinch: str  # "X of Y"
    ground: str  # "X of Y"


class FightDetailPage(BaseModel):
    """Parsed from fight detail page. Full fight with per-round stats."""

    fighter_a_name: str
    fighter_a_url: str
    fighter_b_name: str
    fighter_b_url: str
    fighter_a_status: str  # "W", "L", "D", "NC"
    fighter_b_status: str
    bout_type: str  # "UFC Light Heavyweight Title Bout"
    method: str | None
    method_detail: str | None
    round_finished: int | None
    time_finished: str | None
    time_format: str | None  # "5 Rnd (5-5-5-5-5)"
    referee: str | None
    totals: tuple[RoundStatsRaw, RoundStatsRaw]  # (fighter_a, fighter_b)
    sig_strikes: tuple[SigStrikesRaw, SigStrikesRaw]
    per_round_totals: list[tuple[RoundStatsRaw, RoundStatsRaw]]
    per_round_sig_strikes: list[tuple[SigStrikesRaw, SigStrikesRaw]]


class FighterProfile(BaseModel):
    """Parsed from fighter profile page."""

    name: str
    nickname: str | None = None
    height_str: str | None = None  # "6' 4\"" or "--"
    weight_str: str | None = None  # "205 lbs." or "--"
    reach_str: str | None = None  # "77\"" or "--"
    stance: str | None = None
    dob_str: str | None = None  # "Nov 07, 1990" or ""
