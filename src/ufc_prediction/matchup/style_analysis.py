"""Style analysis: style index and style-vs-style historical win rates.

Computes the striker-vs-grappler index from domain Elo (D-10) and
aggregates historical win rates by style matchup type (D-08, D-09).
"""

from __future__ import annotations

from dataclasses import dataclass

# Minimum fights required to report a win rate (D-09)
MIN_SAMPLE_SIZE = 10


@dataclass
class StyleMatchupWinRate:
    """Win rate for a specific style matchup type.

    style_a and style_b are sorted alphabetically for canonical ordering.
    style_a_win_pct is None when total < MIN_SAMPLE_SIZE (D-09).
    """

    style_a: str
    style_b: str
    style_a_wins: int
    style_b_wins: int
    total: int
    style_a_win_pct: float | None
    sufficient_data: bool


def compute_style_index(
    striking_elo: float | None,
    grappling_elo: float | None,
) -> float | None:
    """Compute striker-vs-grappler index from domain Elo ratings.

    Args:
        striking_elo: Fighter's striking Elo rating, or None.
        grappling_elo: Fighter's grappling Elo rating, or None.

    Returns:
        striking_elo - grappling_elo (D-10). Positive = more striker-oriented,
        negative = more grappler-oriented. None if either Elo is missing.
    """
    if striking_elo is None or grappling_elo is None:
        return None
    return striking_elo - grappling_elo


def compute_style_vs_style_win_rates(
    matchup_counts: dict[tuple[str, str], dict[str, int]],
) -> list[StyleMatchupWinRate]:
    """Compute win rates for each style matchup type.

    Args:
        matchup_counts: Dict mapping (style_a, style_b) tuples (alphabetically
            sorted) to {"a_wins": int, "b_wins": int, "total": int}.
            From queries.get_style_matchup_counts().

    Returns:
        List of StyleMatchupWinRate sorted alphabetically by (style_a, style_b).
        Win percentages are None when total < 10 (D-09).
    """
    results: list[StyleMatchupWinRate] = []

    for (style_a, style_b), counts in matchup_counts.items():
        total = counts["total"]
        a_wins = counts["a_wins"]
        b_wins = counts["b_wins"]
        sufficient = total >= MIN_SAMPLE_SIZE

        results.append(
            StyleMatchupWinRate(
                style_a=style_a,
                style_b=style_b,
                style_a_wins=a_wins,
                style_b_wins=b_wins,
                total=total,
                style_a_win_pct=(a_wins / total) if sufficient else None,
                sufficient_data=sufficient,
            )
        )

    # Sort alphabetically by (style_a, style_b) for canonical order
    results.sort(key=lambda r: (r.style_a, r.style_b))
    return results
