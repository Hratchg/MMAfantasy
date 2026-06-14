"""Opponent-adjusted performance rates.

Divides a fighter's actual stat by the opponent's career average to
produce a rate that shows performance relative to strength of schedule
(D-07, D-08). Values > 1.0 indicate the fighter performed better than
what the opponent typically faces.
"""

from __future__ import annotations


def compute_opponent_adjusted(
    actual: float,
    opponent_career_avg: float | None,
) -> float:
    """Compute opponent-adjusted performance rate.

    Args:
        actual: Fighter's actual stat value for this fight.
        opponent_career_avg: Opponent's career average for this stat
            up to the fight date, or None if unavailable.

    Returns:
        actual / opponent_career_avg. If opponent_career_avg is None
        or 0, returns actual unchanged (zero/None guard per D-07).
    """
    if opponent_career_avg is None or opponent_career_avg == 0:
        return actual

    return actual / opponent_career_avg
