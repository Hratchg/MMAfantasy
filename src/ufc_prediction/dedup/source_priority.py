"""Unified source-priority dedup helper (HOUSE-04, Pitfall #11/#13).

Replaces _prefer_canonical (predictor.py:30-63) and _RANKINGS_SOURCE_PRIORITY
(fighter_queries.py:17-25, 222-238). Both consumers import from here.

Phase 14 (DEDUP-01) priority: ufcstats > sherdog > kaggle-rajeevw > kaggle-mdabbert.
Unknown sources sort last (priority 99). Tie-break: caller-supplied tiebreak key
(predictor uses fighter.id; fighter_queries uses -elo_after_shrinkage).

Per 16-PATTERNS.md Gotcha 3: tiebreak_key callable is REQUIRED to support both
legacy consumers — predictor.py ties on `f.id` (lowest fighter_id wins);
fighter_queries.py ties on `-float(r.elo_after_shrinkage)` (highest Elo wins).
A naive merge that hardcodes one tiebreak silently regresses the other consumer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

T = TypeVar("T")

# Legacy parity: dict literal contents must match _SOURCE_PRIORITY (predictor.py:30-37)
# and _RANKINGS_SOURCE_PRIORITY (fighter_queries.py:17-25) byte-for-byte. The HOUSE-04
# parity test asserts this dict equality so any silent priority reorder trips CI.
SOURCE_PRIORITY: dict[str, int] = {
    "ufcstats": 0,
    "sherdog": 1,
    "kaggle-rajeevw": 2,
    "kaggle-mdabbert": 3,
}


def prefer_canonical(  # noqa: UP047
    rows: Sequence[T],
    *,
    source_key: Callable[[T], str],
    tiebreak_key: Callable[[T], Any] = lambda _r: 0,
) -> T:
    """Return the highest-priority row from a duplicate set.

    Args:
        rows: One or more candidate rows (Fighter records, ranking rows, etc.)
            representing the same logical entity across ingest sources. Must
            be non-empty.
        source_key: Callable extracting the row's ingest source string. The
            value is looked up in SOURCE_PRIORITY (unknown sources sort last
            at priority 99). Required so the helper stays type-agnostic.
        tiebreak_key: Callable extracting the secondary sort key when two
            rows share the same source priority. Caller decides the tiebreak
            domain — predictor.py uses `fighter.id` (deterministic across
            re-runs); fighter_queries.py uses `-float(elo_after_shrinkage)`
            (highest Elo wins inside the same source). Defaults to a
            zero-constant which yields stable but arbitrary order on ties.

    Returns:
        The single canonical row.

    Raises:
        ValueError: If `rows` is empty.
    """
    if not rows:
        raise ValueError("prefer_canonical requires at least one row")
    return min(
        rows,
        key=lambda r: (SOURCE_PRIORITY.get(source_key(r), 99), tiebreak_key(r)),
    )
