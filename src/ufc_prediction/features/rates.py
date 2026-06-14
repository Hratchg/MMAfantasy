"""Per-fight rate feature computation from round stats.

Computes all 8 rate features per D-05 from aggregated round stats:
sig_str_per_minute, total_str_per_minute, td_rate, td_accuracy,
td_defense, strike_defense, ctrl_time_per_fight, sub_att_per_fight.

Fight duration estimated from round_finished/time_finished per RESEARCH Pitfall 3.
"""

from __future__ import annotations


def estimate_fight_duration_seconds(
    round_finished: int | None,
    time_finished: str | None,
    num_rounds: int,
) -> int:
    """Estimate total fight duration in seconds.

    If round_finished and time_finished are both present, computes:
        (round_finished - 1) * 300 + parsed_seconds

    Otherwise falls back to num_rounds * 300.

    Args:
        round_finished: Round in which the fight ended (1-indexed), or None.
        time_finished: Time within the round as "M:SS", or None.
        num_rounds: Total scheduled rounds (typically 3 or 5).

    Returns:
        Estimated fight duration in seconds.
    """
    if round_finished is not None and time_finished is not None:
        try:
            parts = time_finished.split(":")
            minutes = int(parts[0])
            seconds = int(parts[1])
            return (round_finished - 1) * 300 + minutes * 60 + seconds
        except (ValueError, IndexError):
            pass  # fall through to default

    return num_rounds * 300


def compute_rate_features(
    fighter_round_stats: list[dict[str, int | None]],
    opponent_round_stats: list[dict[str, int | None]],
    fight_duration_seconds: int,
) -> dict[str, float | None]:
    """Compute per-fight rate features from aggregated round stats.

    Aggregates across all rounds for the fighter in this fight, then
    computes rates. Division-by-zero is guarded with None returns.

    Args:
        fighter_round_stats: List of per-round stat dicts for the fighter.
        opponent_round_stats: List of per-round stat dicts for the opponent.
        fight_duration_seconds: Estimated fight duration in seconds.

    Returns:
        Dict with all 8 rate feature keys per D-05. Values may be None
        when division by zero would occur.
    """

    def _sum_stat(stats: list[dict[str, int | None]], key: str) -> int:
        """Sum a stat across rounds, treating None as 0."""
        return sum(r.get(key) or 0 for r in stats)

    # Aggregate fighter stats across rounds
    sig_str_landed = _sum_stat(fighter_round_stats, "sig_strikes_landed")
    head_landed = _sum_stat(fighter_round_stats, "head_strikes_landed")
    body_landed = _sum_stat(fighter_round_stats, "body_strikes_landed")
    leg_landed = _sum_stat(fighter_round_stats, "leg_strikes_landed")
    td_landed = _sum_stat(fighter_round_stats, "takedowns_landed")
    td_attempted = _sum_stat(fighter_round_stats, "takedowns_attempted")
    sub_attempts = _sum_stat(fighter_round_stats, "submission_attempts")
    ctrl_time = _sum_stat(fighter_round_stats, "control_time_seconds")

    # Aggregate opponent stats for defensive metrics
    opp_sig_str_landed = _sum_stat(opponent_round_stats, "sig_strikes_landed")
    opp_sig_str_attempted = _sum_stat(opponent_round_stats, "sig_strikes_attempted")
    opp_td_landed = _sum_stat(opponent_round_stats, "takedowns_landed")
    opp_td_attempted = _sum_stat(opponent_round_stats, "takedowns_attempted")

    # Total strikes from positional breakdown (no total_strikes column per RESEARCH)
    total_strikes_landed = head_landed + body_landed + leg_landed

    # Per-minute rates (T-07-01: guard division by zero)
    fight_minutes = fight_duration_seconds / 60.0 if fight_duration_seconds > 0 else 0.0

    sig_str_per_minute: float | None = None
    total_str_per_minute: float | None = None
    if fight_minutes > 0:
        sig_str_per_minute = sig_str_landed / fight_minutes
        total_str_per_minute = total_strikes_landed / fight_minutes

    # Takedown accuracy (T-07-01: None if no attempts)
    td_accuracy: float | None = None
    if td_attempted > 0:
        td_accuracy = td_landed / td_attempted

    # Takedown defense (T-07-01: None if opponent had no attempts)
    td_defense: float | None = None
    if opp_td_attempted > 0:
        td_defense = 1.0 - (opp_td_landed / opp_td_attempted)

    # Strike defense (T-07-01: None if opponent had no attempts)
    strike_defense: float | None = None
    if opp_sig_str_attempted > 0:
        strike_defense = 1.0 - (opp_sig_str_landed / opp_sig_str_attempted)

    return {
        "sig_str_per_minute": sig_str_per_minute,
        "total_str_per_minute": total_str_per_minute,
        "td_rate": td_landed,  # raw count per fight, not per minute
        "td_accuracy": td_accuracy,
        "td_defense": td_defense,
        "strike_defense": strike_defense,
        "ctrl_time_per_fight": ctrl_time,  # raw total for the fight
        "sub_att_per_fight": sub_attempts,  # raw total for the fight
    }
