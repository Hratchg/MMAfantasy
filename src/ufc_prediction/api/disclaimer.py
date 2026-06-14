"""Shared entertainment-disclaimer constants for HYGIENE-V24-02.

The same 200-word body is surfaced in three places (per CONTEXT.md
``<decisions>`` "Entertainment Disclaimer"):

1. ``README.md`` section ``## Disclaimer`` (Plan 38-03 quotes
   ``DISCLAIMER_200W`` verbatim).
2. ``FastAPI(description=DISCLAIMER_200W)`` -- partners reading
   ``/docs`` or ``/redoc`` (dev/staging only; prod 404s those
   routes) see the disclaimer above the route table.
3. ``PredictorOutputV1.disclaimer`` -- every ``/api/v1/predict``
   response carries the disclaimer as a top-level field
   (additive to v1.2.0; Phase 25 forward-compat lock still binds --
   see Plan 38-02 Task 4 / CONTRACT-V24-03 re-verification).

Word count is in the 200 +/- 20 closed range per CONTEXT.md
``<decisions>`` "Entertainment Disclaimer -> Content". The constant
is plain text (no Markdown headings); whitespace is preserved.
"""

from __future__ import annotations

DISCLAIMER_200W: str = (
    "UFC Fight Prediction is a fantasy-MMA decision-support tool "
    "intended to help fantasy players reason about matchups using "
    "historical performance data. For informational and "
    "entertainment purposes; not financial or wagering advice; "
    "statistical estimates may diverge from actual outcomes. The "
    "win-probability values, baseline comparisons, and calibration "
    "metrics surfaced by this API are machine-learning estimates "
    "trained on historical UFC fight data through May 2026. They "
    "are NOT predictions of future events, NOT guarantees of any "
    "particular outcome, and NOT a recommendation to place, modify, "
    "or refrain from any wager, fantasy pick, or other transaction. "
    "Model uncertainty is real and material: closing-odds-derived "
    "features can be missing or stale at predict time, sample sizes "
    "for debutants and short-tenured fighters are small, and the "
    "underlying corpus is incomplete in well-documented ways "
    "(significant-strike classification noise, referee scoring "
    "variance, late-injury substitutions, weight-cut anomalies, "
    "and judging idiosyncrasies). Partners integrating this API "
    "into a downstream product are solely responsible for assessing "
    "fitness for purpose, complying with local laws regarding "
    "fantasy contests, sports wagering, and games of skill, and "
    "clearly framing model outputs to their end users as statistical "
    "estimates rather than outcome predictions or guaranteed "
    "results. Nothing in this API constitutes regulated financial "
    "advice, tax guidance, sports-betting recommendations, legal "
    "counsel, or professional consultation of any kind, and use of "
    "the API does not establish any advisory or fiduciary relationship."
)
