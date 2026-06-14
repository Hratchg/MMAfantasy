"""v2.2 META rich Level-1 features (Phase 23 Plan 23-03 — D-06).

All features DERIVED from existing tables — no new scrapers, no new
migrations. Pre-fight as-of-date discipline enforced via
``division_history`` filter (strict ``event_date < as_of_date`` cutoff).
Beta-binomial EB shrinkage helper REUSED from ``features_v22.ref`` (DRY
per CONTEXT D-06 bullet "Beta-binomial EB shared helper").

Both ``feature_matrix.py`` (training path) and ``inference_features.py``
(live-prediction path) import from this module so train/inference math
is byte-identical (Pitfall #12 LIVE-03 parity per CONTEXT D-10).

DEDUP DISCIPLINE per CONTEXT D-07 + RESEARCH Q4 RESOLVED: NO
``elo_*_diff`` / ``age_diff`` / ``stance_matchup`` / raw ``reach_diff``
/ ``layoff_days_diff`` — all duplicates of cols already in
``FEATURE_COLUMNS_NO_NET``. In particular ``layoff_days_diff`` was
DROPPED because ``days_since_last_fight_diff`` at
``FEATURE_COLUMNS_NO_NET[61]`` is semantically identical. This module
adds ONLY the new velocity / division-prior / age-absolute /
layoff-clipped-per-fighter / reach-normalized cols.

Public helpers:
    - ``layoff_days(current, prior)``: per-fighter days since last
      fight, clipped at 720 (D-06). First fight returns 0.
    - ``age_at_fight(dob, fight_date)``: age in years using
      ``fight_date`` (NOT ``date.today()`` — Pitfall #5).
    - ``elo_velocity(history, window=5)``: rate-of-change over N prior
      fights. Insufficient history returns NaN.
    - ``division_finish_rate_shrunk(...)``: Beta-binomial EB per
      weight-class finish rate. Reuses ``ref.beta_binomial_shrink``.
    - ``reach_diff_normalized(reach_a, reach_b, mean)``: reach diff
      scaled by mean reach in division. None inputs / zero mean → NaN.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from ufc_prediction.ml.features_v22.ref import (
    beta_binomial_shrink,
    classify_outcome,
)


def layoff_days(
    current_fight_date: date,
    prior_fight_date: date | None,
    *,
    clip_max: int = 720,
) -> float:
    """Days since fighter's last fight, clipped at ``clip_max`` (D-06).

    First-fight fighter (``prior_fight_date is None``) → 0.0 (NOT NaN;
    symmetric with travel-sentinel semantics — a debutant has 0 layoff,
    not unknown).

    Clip BEFORE the diff (Pitfall #8): per-fighter clip.

    IMPORTANT: This helper is consumed for the raw per-fighter
    ``layoff_days_red`` / ``layoff_days_blue`` cols ONLY. The
    differential form ``layoff_days_diff`` is NOT emitted —
    ``days_since_last_fight_diff`` already exists at
    ``FEATURE_COLUMNS_NO_NET[61]`` (Q4 RESOLVED + D-07).

    Args:
        current_fight_date: The fight whose row is being computed.
        prior_fight_date: This fighter's most-recent prior fight, or
            ``None`` for a debutant.
        clip_max: Maximum days to clip; default 720 per D-06.

    Returns:
        Days as float in [0, ``clip_max``]. Negative diffs (impossible
        when caller respects pre-fight discipline) are floored at 0.
    """
    if prior_fight_date is None:
        return 0.0
    days = (current_fight_date - prior_fight_date).days
    return float(min(max(days, 0), clip_max))


def age_at_fight(birth_date: date | None, fight_date: date) -> float:
    """Age in years at ``fight_date``.

    Pitfall #5: uses ``fight_date`` (NOT ``date.today()``) — caller MUST
    pass the event date for the row being computed. Returns NaN if
    ``birth_date`` is unknown (Pattern D).

    Args:
        birth_date: Fighter's date of birth, nullable in the data
            (incomplete fighter records).
        fight_date: The event date for which age is being computed.

    Returns:
        Age in years as ``(fight_date - birth_date).days / 365.25``.
        ``float('nan')`` when ``birth_date is None``.
    """
    if birth_date is None:
        return float("nan")
    return (fight_date - birth_date).days / 365.25


def elo_velocity(elo_history: list[float], window: int = 5) -> float:
    """Per-fight Elo rate-of-change: ``(current - elo_N_fights_ago) / N``.

    Insufficient history (``len < window + 1``) → NaN per Pattern D.
    Default window N=5 (D-06 planner discretion; v2.3+ may tune per
    Phase 26 META signal-sensitivity).

    Args:
        elo_history: Chronological list of Elo values for one fighter,
            most recent LAST. To compute velocity at the time of a fight,
            pass the PRIOR-only history (strict ``event_date < as_of``).
        window: Lookback distance N; need ``N + 1`` total entries
            (current value + N priors).

    Returns:
        ``(elo_history[-1] - elo_history[-(window + 1)]) / window`` when
        sufficient history; ``float('nan')`` otherwise.
    """
    if len(elo_history) < window + 1:
        return float("nan")
    return (elo_history[-1] - elo_history[-(window + 1)]) / window


def division_finish_rate_shrunk(
    weight_class: str | None,
    as_of_date: date,
    division_history: dict[str, list[dict[str, Any]]],
    global_finish_rate: float,
    k_shrink: float = 50.0,
) -> float:
    """Per-division finish rate with Beta-binomial EB shrinkage.

    REUSES ``ref.beta_binomial_shrink`` + ``ref.classify_outcome`` (DRY
    per CONTEXT D-06). Strict pre-fight as-of-date discipline (Pitfall
    #4): only fights with ``event_date < as_of_date`` contribute.

    Unknown class (``weight_class is None`` or not in
    ``division_history``) or no prior history → collapse to
    ``global_finish_rate`` via ``beta_binomial_shrink(0, 0, g, k) == g``.

    Args:
        weight_class: Division key (e.g. "Lightweight"). ``None``
            triggers Bayesian fallback to ``global_finish_rate``.
        as_of_date: Strict pre-fight cutoff. Only entries with
            ``event_date < as_of_date`` contribute.
        division_history: ``{weight_class: [{"event_date": d,
            "method": m}, ...]}`` — chronological list per division.
        global_finish_rate: Pool rate across the full corpus; used as
            the Beta prior.
        k_shrink: Shrinkage strength. Default 50 (D-02 rule of thumb).

    Returns:
        Posterior mean finish rate as float in [0, 1].
    """
    if weight_class is None or weight_class not in division_history:
        return beta_binomial_shrink(0, 0, global_finish_rate, k_shrink)
    finishes = 0
    n = 0
    for f in division_history[weight_class]:
        if f["event_date"] >= as_of_date:
            continue  # strict pre-fight cutoff (Pitfall #4)
        n += 1
        if classify_outcome(f.get("method")) == "finish":
            finishes += 1
    return beta_binomial_shrink(finishes, n, global_finish_rate, k_shrink)


def reach_diff_normalized(
    reach_a: float | None,
    reach_b: float | None,
    mean_reach_in_division: float,
) -> float:
    """``(reach_a - reach_b) / mean_reach_in_division``. NaN on miss.

    Distinct from existing ``FEATURE_COLUMNS_NO_NET`` ``reach_diff`` (raw):
    this col is divided by mean reach in the fighters' weight class,
    controlling for division-level reach distribution.

    Either-None or zero-mean → NaN (Pattern D + div-by-zero guard).

    Args:
        reach_a: Red fighter's reach in inches; ``None`` when missing.
        reach_b: Blue fighter's reach in inches; ``None`` when missing.
        mean_reach_in_division: Mean reach across the fighter pool for
            this weight class. Zero → NaN (avoid div-by-zero).

    Returns:
        Normalized reach difference; ``float('nan')`` on any missing
        input or zero denominator.
    """
    if reach_a is None or reach_b is None or mean_reach_in_division == 0.0:
        return float("nan")
    return (reach_a - reach_b) / mean_reach_in_division
