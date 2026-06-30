"""REF (referee) features for v2.2 — Beta-binomial EB shrinkage.

Phase 23 REF-V22-02. Three per-event REF cols (NOT per-fighter-differential
per CONTEXT D-03):
    - ref_finish_rate_shrunk
    - ref_decision_rate_shrunk
    - ref_no_action_rate_shrunk

All math is pure (no DB session). Both the training path
(``feature_matrix.py``) and the live-prediction path
(``inference_features.py``) import these helpers so REF compute is
byte-identical (Pitfall #12 LIVE-03 parity per CONTEXT D-10).

Strict pre-fight discipline: ``compute_ref_rates_shrunk`` filters
``ref_history`` entries by ``event_date < as_of_date`` (Pitfall #4
leakage guard).

Beta-binomial closed-form posterior mean per CONTEXT D-02:
    posterior_mean = (successes + k_shrink * global_rate) / (n + k_shrink)

Default ``k_shrink = 50.0`` is the D-02 rule-of-thumb literal. CV-tuning
of ``k_shrink`` is BANNED in Phase 23 (AF-1 starts Phase 24).
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal


# ── Method-string → category map (Q2 RESOLVED) ──────────────────────────────

_FINISH_METHODS: frozenset[str] = frozenset(
    {
        "KO/TKO",
        "TKO",
        "KO",
        "Submission",
        "SUB",
    }
)

_DECISION_METHODS: frozenset[str] = frozenset(
    {
        "Decision",
        "Decision - Unanimous",
        "Decision - Split",
        "Decision - Majority",
    }
)

# All other strings (No Contest, DQ, Disqualification, Overturned, None,
# unknown) collapse to "no_action" — defensive default per Q2 RESOLVED.

OutcomeCategory = Literal["finish", "decision", "no_action"]


def classify_outcome(method: str | None) -> OutcomeCategory:
    """Classify a fight's ``method`` string into one of 3 REF categories.

    - ``finish``: KO / TKO / Submission / SUB
    - ``decision``: any Decision variant
    - ``no_action``: No Contest / DQ / Disqualification / Overturned / None /
      any unknown method string (defensive default per Q2 RESOLVED).

    No exception is raised for unknown method strings — they collapse to
    ``no_action`` so a malformed DB row cannot crash REF compute.
    """
    if method is None:
        return "no_action"
    if method in _FINISH_METHODS:
        return "finish"
    if method in _DECISION_METHODS:
        return "decision"
    return "no_action"


# ── Beta-binomial EB shrinkage helper ───────────────────────────────────────


def beta_binomial_shrink(
    successes: int,
    n: int,
    global_rate: float,
    k_shrink: float = 50.0,
) -> float:
    """Closed-form Beta-binomial empirical-Bayes posterior mean.

    posterior_mean = (successes + k_shrink * global_rate) / (n + k_shrink)

    - ``successes``: count of events in the target category for this referee.
    - ``n``: total events for this referee (pre-fight as-of-date).
    - ``global_rate``: pool rate across the full corpus, used as the prior.
    - ``k_shrink``: shrinkage strength (50 = rule-of-thumb per D-02).

    When ``n == 0`` (unseen referee), returns exactly ``global_rate`` — the
    Bayesian-fallback discipline of D-02 (Pitfall #2: no zero-division).

    Output is bounded ``[0, 1]`` as long as ``0 ≤ successes ≤ n`` and
    ``0 ≤ global_rate ≤ 1``.
    """
    return (successes + k_shrink * global_rate) / (n + k_shrink)


# ── Per-referee compute ─────────────────────────────────────────────────────


def compute_ref_rates_shrunk(
    referee_id: int | None,
    as_of_date: date,
    ref_history: dict[int, list[dict[str, Any]]],
    global_rates: dict[str, float],
    k_shrink: float = 50.0,
) -> dict[str, float]:
    """Compute the 3 REF cols for a single (referee, as_of_date) pair.

    Args:
        referee_id: Referee FK. ``None`` triggers the Bayesian fallback
            (all 3 cols == ``global_rates`` for that category).
        as_of_date: Strict pre-fight cutoff. Only history entries with
            ``event_date < as_of_date`` contribute (Pitfall #4).
        ref_history: ``{referee_id: [{"event_date": d, "method": m}, ...]}``.
            Missing referee_id triggers the Bayesian fallback.
        global_rates: ``{"finish": x, "decision": y, "no_action": z}`` —
            pool rates across the full corpus, used as the prior.
        k_shrink: Shrinkage strength. Default 50 (D-02 rule of thumb).

    Returns:
        Dict with 3 keys: ``ref_finish_rate_shrunk``,
        ``ref_decision_rate_shrunk``, ``ref_no_action_rate_shrunk``.

    Bayesian fallback discipline: when the referee is unknown OR has no
    prior history at ``as_of_date``, all 3 outputs are ``global_rates[cat]``
    (i.e. ``beta_binomial_shrink(0, 0, g, k) == g``). Pitfall #2 / Pitfall #5.
    """
    # Bayesian fallback for unknown referees.
    if referee_id is None or referee_id not in ref_history:
        return {
            "ref_finish_rate_shrunk": global_rates["finish"],
            "ref_decision_rate_shrunk": global_rates["decision"],
            "ref_no_action_rate_shrunk": global_rates["no_action"],
        }

    counts = {"finish": 0, "decision": 0, "no_action": 0}
    n = 0
    for past in ref_history[referee_id]:
        # Strict less-than cutoff (Pitfall #4 — same-day fights are
        # excluded to avoid bleeding the current event's own outcome).
        if past["event_date"] >= as_of_date:
            continue
        cat = classify_outcome(past.get("method"))
        counts[cat] += 1
        n += 1

    return {
        "ref_finish_rate_shrunk": beta_binomial_shrink(
            counts["finish"],
            n,
            global_rates["finish"],
            k_shrink,
        ),
        "ref_decision_rate_shrunk": beta_binomial_shrink(
            counts["decision"],
            n,
            global_rates["decision"],
            k_shrink,
        ),
        "ref_no_action_rate_shrunk": beta_binomial_shrink(
            counts["no_action"],
            n,
            global_rates["no_action"],
            k_shrink,
        ),
    }
