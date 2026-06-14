"""Fixtures and helpers for Elo engine unit tests."""

from __future__ import annotations

from datetime import date

import pytest

from ufc_prediction.elo.config import EloConfig
from ufc_prediction.elo.engine import EloEngine, FightRecord

_fight_id_counter = 0


def make_fight(
    *,
    fight_id: int | None = None,
    event_date: date = date(2020, 1, 1),
    fighter_a_id: int = 1,
    fighter_b_id: int = 2,
    winner_id: int | None = 1,
    weight_class: str = "Lightweight",
    method: str | None = "Decision",
    method_detail: str | None = "Unanimous",
) -> FightRecord:
    """Create a FightRecord with sensible defaults, auto-incrementing fight_id."""
    global _fight_id_counter
    if fight_id is None:
        _fight_id_counter += 1
        fight_id = _fight_id_counter
    return FightRecord(
        fight_id=fight_id,
        event_date=event_date,
        fighter_a_id=fighter_a_id,
        fighter_b_id=fighter_b_id,
        winner_id=winner_id,
        weight_class=weight_class,
        method=method,
        method_detail=method_detail,
    )


@pytest.fixture(autouse=True)
def _reset_fight_id_counter() -> None:
    """Reset the fight_id counter before each test."""
    global _fight_id_counter
    _fight_id_counter = 0


def make_round_stats(
    *,
    fight_id: int,
    fighter_id: int,
    sig_str_landed: int = 20,
    takedowns_landed: int = 1,
    ctrl_time_seconds: int = 30,
) -> dict:
    """Create a round stats dict with sensible defaults for domain Elo tests."""
    return {
        "fighter_id": fighter_id,
        "sig_str_landed": sig_str_landed,
        "takedowns_landed": takedowns_landed,
        "ctrl_time_seconds": ctrl_time_seconds,
    }


@pytest.fixture()
def default_config() -> EloConfig:
    """Return an EloConfig with all defaults."""
    return EloConfig()


@pytest.fixture()
def elo_engine(default_config: EloConfig) -> EloEngine:
    """Return a fresh EloEngine with default config."""
    return EloEngine(default_config)
