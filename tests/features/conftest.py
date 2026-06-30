"""Shared fixtures for feature computation tests."""

from __future__ import annotations


def make_round_stats(overrides: dict | None = None) -> dict:
    """Factory returning a dict with all RoundStats column keys.

    Defaults to reasonable non-None values. Merges overrides.
    """
    base = {
        "sig_strikes_landed": 5,
        "sig_strikes_attempted": 10,
        "head_strikes_landed": 2,
        "body_strikes_landed": 1,
        "leg_strikes_landed": 1,
        "distance_strikes_landed": 2,
        "clinch_strikes_landed": 1,
        "ground_strikes_landed": 1,
        "takedowns_landed": 1,
        "takedowns_attempted": 2,
        "submission_attempts": 0,
        "control_time_seconds": 30,
        "knockdowns": 0,
        "reversals": 0,
    }
    if overrides:
        base.update(overrides)
    return base


def make_fighter_round_stats(
    fighter_id: int,
    fight_id: int,
    rounds: int = 3,
    overrides: dict | None = None,
) -> list[dict]:
    """Return a list of per-round stat dicts for a fighter in a fight.

    Each dict has fighter_id, fight_id, round_number set, plus
    make_round_stats(overrides) values.
    """
    result = []
    for r in range(1, rounds + 1):
        stats = make_round_stats(overrides)
        stats["fighter_id"] = fighter_id
        stats["fight_id"] = fight_id
        stats["round_number"] = r
        result.append(stats)
    return result
