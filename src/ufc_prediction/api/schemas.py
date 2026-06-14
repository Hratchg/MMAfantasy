"""Pydantic response models for all API endpoints."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel

# ── Fighter Ratings (API-01) ────────────────────────────────────────────


class FighterSearchResult(BaseModel):
    """Single fighter candidate returned when multiple matches exist."""

    name: str
    id: int
    division: str | None = None
    elo: float | None = None


class FighterSearchResponse(BaseModel):
    """Returned when a name search matches multiple fighters (D-13)."""

    count: int
    results: list[FighterSearchResult]


class DivisionRating(BaseModel):
    """Rating detail for one division a fighter competes in."""

    division: str
    elo: float | None = None
    striking_elo: float | None = None
    grappling_elo: float | None = None
    wins: int = 0
    losses: int = 0
    total_fights: int = 0
    last_fight_date: date | None = None


class FighterRatingResponse(BaseModel):
    """Full fighter rating details (single match)."""

    name: str
    id: int
    divisions: list[DivisionRating]


# ── Division Rankings (API-02) ──────────────────────────────────────────


class RankedFighter(BaseModel):
    """A fighter entry within a division ranking."""

    rank: int
    name: str
    fighter_id: int
    elo: float
    fights: int
    last_active: date


class RankingsResponse(BaseModel):
    """Ranked list of fighters within a division."""

    division: str
    fighters: list[RankedFighter]


# ── Matchup Comparison (API-03) ─────────────────────────────────────────


class PhysicalModel(BaseModel):
    """Physical attribute differentials between two fighters."""

    height_diff: float | None = None
    reach_diff: float | None = None
    leg_reach_diff: float | None = None
    age_diff: float | None = None
    fighter_a_age: float | None = None
    fighter_b_age: float | None = None


class EloComparisonModel(BaseModel):
    """Elo comparison across overall and domain ratings."""

    fighter_a_overall: float | None = None
    fighter_b_overall: float | None = None
    fighter_a_striking: float | None = None
    fighter_b_striking: float | None = None
    fighter_a_grappling: float | None = None
    fighter_b_grappling: float | None = None
    fighter_a_division: str | None = None
    fighter_b_division: str | None = None


class StrikingModel(BaseModel):
    """Striking matchup matrix."""

    a_sig_str_per_min: float | None = None
    a_striking_accuracy: float | None = None
    a_strike_defense: float | None = None
    b_sig_str_per_min: float | None = None
    b_striking_accuracy: float | None = None
    b_strike_defense: float | None = None


class GrapplingModel(BaseModel):
    """Grappling matchup matrix."""

    a_td_rate: float | None = None
    a_td_accuracy: float | None = None
    a_td_defense: float | None = None
    b_td_rate: float | None = None
    b_td_accuracy: float | None = None
    b_td_defense: float | None = None
    a_ctrl_time: float | None = None
    b_ctrl_time: float | None = None


class StyleMatchupWinRateModel(BaseModel):
    """Win rate for a specific style matchup type."""

    style_a: str
    style_b: str
    style_a_wins: int
    style_b_wins: int
    total: int
    style_a_win_pct: float | None = None
    sufficient_data: bool


class StyleModel(BaseModel):
    """Style analysis section of a matchup comparison."""

    fighter_a_style: str | None = None
    fighter_b_style: str | None = None
    fighter_a_index: float | None = None
    fighter_b_index: float | None = None
    matchup_win_rates: list[StyleMatchupWinRateModel]
    cross_division_warning: str | None = None


class MatchupResponse(BaseModel):
    """Full matchup comparison between two fighters."""

    fighter_a_name: str
    fighter_b_name: str
    physical: PhysicalModel
    elo: EloComparisonModel
    striking: StrikingModel
    grappling: GrapplingModel
    style: StyleModel


# ── Elo History (API-04) ────────────────────────────────────────────────


class EloHistoryPoint(BaseModel):
    """A single point in a fighter's Elo trajectory."""

    fight_date: date
    division: str
    elo_before: float
    elo_after: float
    elo_after_shrinkage: float


class EloHistoryResponse(BaseModel):
    """Chronological Elo history for a fighter."""

    name: str
    fighter_id: int
    history: list[EloHistoryPoint]
