"""BestFightOdds vig-removal and implied-probability math.

Pure functions — no I/O, no DB, no third-party scraping deps.
All formulas cited from .planning/phases/13-betting-odds-integration/13-RESEARCH.md
section "Pattern 2: Vig removal via proportional normalization" and the
D-02 locked decision in 13-CONTEXT.md.

Decisions implemented:
- D-01: closing consensus via book-to-book range midpoint (`closing_ml_consensus`)
- D-02: vig removed via proportional / multiplicative normalization (`devig_proportional`)
- D-03: the caller preserves the raw American ML alongside the computed probability
"""

from __future__ import annotations


def american_ml_to_implied_prob(ml: int) -> float:
    """Raw implied probability from an American moneyline (includes vig).

    Per D-03: the caller preserves the raw ML alongside this computed prob.

    Formulas (cited in RESEARCH.md Pattern 2):
      ml < 0  (favorite): abs(ml) / (abs(ml) + 100)
      ml >= 0 (underdog): 100 / (ml + 100)

    Examples:
      -200 -> 0.6667 (2/3)
      +170 -> 0.3704 (100/270)
      +100 -> 0.5 (pickem)
    """
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    return 100 / (ml + 100)


def devig_proportional(ml_a: int, ml_b: int) -> tuple[float, float]:
    """Remove vig by normalizing two implied probabilities to sum=1.0 (D-02).

    Worked example (matches D-02 comment / RESEARCH.md Pattern 2):
      ML_A=-200 (raw p=0.6667), ML_B=+170 (raw p=0.3704).
      Raw sum = 1.0370 (3.7% vig).
      Normalized: p_A=0.6429, p_B=0.3571.

    Returns:
      (prob_a, prob_b) both in [0.0, 1.0], summing to 1.0 within fp precision.
    """
    p_a = american_ml_to_implied_prob(ml_a)
    p_b = american_ml_to_implied_prob(ml_b)
    total = p_a + p_b
    return p_a / total, p_b / total


def closing_ml_consensus(min_ml: int, max_ml: int) -> int:
    """Midpoint of the book-to-book closing range (D-01 consensus).

    Closing odds are stored as a RANGE per RESEARCH.md (ufcscraper records
    min/max across books); we collapse to a single midpoint before computing
    the implied probability. Uses Python 3 banker's rounding via ``round``
    (half-to-even) with no ``ndigits`` argument — returns ``int`` directly.

    Example:
      (-200, -180) -> -190
      (-110, 120)  -> 5
    """
    return round((min_ml + max_ml) / 2)
