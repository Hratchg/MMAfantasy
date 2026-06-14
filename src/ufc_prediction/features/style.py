"""Style classification from domain Elo differential.

Classifies fighters as striker, grappler, or balanced based on the
difference between their striking and grappling Elo ratings (D-01).
"""

from __future__ import annotations


def classify_style(
    striking_elo: float | None,
    grappling_elo: float | None,
    threshold: float = 30.0,
) -> str:
    """Classify a fighter's style from domain Elo differential.

    Args:
        striking_elo: Fighter's current striking Elo rating, or None.
        grappling_elo: Fighter's current grappling Elo rating, or None.
        threshold: Minimum absolute differential for non-balanced tag.
            Default 30.0 Elo points (tunable per D-01).

    Returns:
        "striker" if striking_elo - grappling_elo > threshold,
        "grappler" if striking_elo - grappling_elo < -threshold,
        "balanced" otherwise (including when either Elo is None).
    """
    if striking_elo is None or grappling_elo is None:
        return "balanced"

    diff = striking_elo - grappling_elo

    if diff > threshold:
        return "striker"
    if diff < -threshold:
        return "grappler"
    return "balanced"
