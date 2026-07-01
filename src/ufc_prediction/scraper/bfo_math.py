"""BestFightOdds vig-removal and implied-probability math.

Pure functions — no I/O, no DB, no third-party scraping deps.
All formulas cited from .planning/phases/13-betting-odds-integration/13-RESEARCH.md
section "Pattern 2: Vig removal via proportional normalization" and the
D-02 locked decision in 13-CONTEXT.md.

Decisions implemented:
- D-01: closing consensus across the book-to-book range. NOTE: the original
  D-01 method (midpoint of the American-ML range, `closing_ml_consensus`) was
  mathematically broken for ranges straddling zero and has been replaced by a
  probability-space consensus (`closing_prob_consensus` / `devig_closing_range`).
  Operator: record a superseding D-{N} row for the method change.
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

    Domain guard: American moneylines are discontinuous at 0 — valid values are
    ``ml <= -100`` or ``ml >= 100`` (both -100 and +100 mean even). Values in the
    open interval (-100, 100) are not valid odds; feeding one (e.g. a midpoint
    that averaged two MLs across zero) yields a nonsensical probability, so we
    reject it loudly rather than return garbage. See ``closing_prob_consensus``.
    """
    if -100 < ml < 100:
        raise ValueError(
            f"invalid American moneyline {ml!r}: |ml| must be >= 100 "
            "(values in (-100, 100) are not valid odds — did you average MLs across zero?)"
        )
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


def closing_prob_consensus(min_ml: int, max_ml: int) -> float:
    """Consensus closing implied probability (WITH vig) for one fighter side.

    Closing odds are stored as a RANGE per RESEARCH.md (ufcscraper records
    min/max across books). Consensus is taken in PROBABILITY space — the mean
    of each book's implied probability — NOT by averaging the American
    moneylines.

    This supersedes the original D-01 "midpoint of the ML range" method, which
    was mathematically broken: American MLs are discontinuous at 0 (they jump
    -100 <-> +100 with no valid values between), so any range straddling zero —
    exactly the near-even / pick-'em fights — collapsed to a near-zero midpoint
    that mapped to a nonsensical probability (e.g. (-105, 100) -> midpoint -2 ->
    p=0.0196). Averaging in probability space is continuous and correct.

    Example:
      (-200, -180) -> mean(0.6667, 0.6429) = 0.6548
      (-105,  100) -> mean(0.5122, 0.5000) = 0.5061   (sane; was 0.0196)
    """
    return (american_ml_to_implied_prob(min_ml) + american_ml_to_implied_prob(max_ml)) / 2.0


def devig_closing_range(
    this_min: int, this_max: int, other_min: int, other_max: int
) -> tuple[float, float]:
    """Devigged (this, other) CLOSING probabilities from each side's book range.

    Takes the per-side probability-space consensus (``closing_prob_consensus``)
    then removes vig by proportional normalization (D-02). This is the single
    canonical closing-odds computation — both ``bfo_ingest`` (stored features)
    and ``inference_features`` (serve-time features) MUST call it so the two
    paths cannot drift.

    Returns:
      (prob_this, prob_other) both in [0.0, 1.0], summing to 1.0.
    """
    p_this = closing_prob_consensus(this_min, this_max)
    p_other = closing_prob_consensus(other_min, other_max)
    total = p_this + p_other
    return p_this / total, p_other / total
