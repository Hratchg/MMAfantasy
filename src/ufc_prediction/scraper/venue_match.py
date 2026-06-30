"""Three-tier venue matcher (exact → token_sort_ratio>=85 → unresolved).

Phase 28 INGEST-V23-02. Consumed by ``ufc scrape venues`` CLI to backfill
``events.venue_id`` against the post-Plan-28-01 ``data/venues.csv`` (174 rows
covering every distinct ``events.location`` string in the corpus).

**Why token_sort_ratio (not fuzz.ratio)** — CONTEXT D-06: location strings
exhibit word-order variation across UFCStats / Sherdog / mdabbert ingest
paths (e.g. "Las Vegas, Nevada, USA" vs "Las Vegas, USA, Nevada"). The
fighter-name matcher (``bfo_matcher.match_bfo_name``) uses ``fuzz.ratio``
because reorderings there ("Jon Jones" vs "Jones Jon") are ambiguous and we
prefer the conservative scorer; venues are unambiguous on reordering and
``token_sort_ratio`` gives 100 for equivalent token bags.

**CR-01 idempotency** — the ``assign_venue_id_if_null`` helper carries
Phase 22's set-once policy from referees forward to venues: once
``event.venue_id`` is non-NULL it is never overwritten. Re-runs that
encounter a different (or NULL) match for an already-resolved event are
no-ops, preserving the prior FK.

Banned imports per Pitfall #1 / Finding 11: nothing under
``ufc_prediction.ml.*`` (LIVE-03 + AUDIT-01 byte-identity guard).
"""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz
from sqlalchemy.orm import Session

from ufc_prediction.models.event import Event


_WHITESPACE_RE = re.compile(r"\s+")
_DEFAULT_THRESHOLD: int = 85


def _normalize_location(raw: str) -> str:
    """NFKD-fold + lowercase + collapse internal whitespace + strip.

    Matches the upstream normalization used in ``referee_normalize.py``
    (Phase 22) for consistency with the rest of the scraper layer. Returns
    an empty string for empty / whitespace-only input.
    """
    folded = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    return _WHITESPACE_RE.sub(" ", folded.lower().strip())


def match_venue(
    location: str,
    venue_lookup: dict[str, int],
    venue_names_raw: list[tuple[int, str]],
    threshold: int = _DEFAULT_THRESHOLD,
) -> tuple[int, str] | None:
    """Three-tier match: exact -> rapidfuzz.token_sort_ratio>=threshold -> None.

    Args:
        location: Raw ``events.location`` string (operator-trusted; comes
            from the parsed event detail page).
        venue_lookup: Pre-built mapping ``_normalize_location(name) -> venue_id``
            for O(1) exact lookup. Build once at CLI start from
            ``data/venues.csv``.
        venue_names_raw: List of ``(venue_id, raw_name)`` tuples for the
            fuzzy fallback pass. Iterated in order; the highest-scoring
            candidate above ``threshold`` wins (ties broken by insertion
            order, mirroring ``Counter.most_common`` semantics).
        threshold: Minimum ``token_sort_ratio`` (0-100) to accept a fuzzy
            match. Default 85 per CONTEXT D-06.

    Returns:
        ``(venue_id, match_kind)`` where ``match_kind`` is either
        ``"exact"`` or ``"fuzzy:NN"`` (NN is the integer score). Returns
        ``None`` if no candidate scores at or above ``threshold`` —
        operator surfaces these via ``28-UNMATCHED-VENUES.md`` for manual
        review and CSV addition (no silent guessing per D-09(P15)).
    """
    norm = _normalize_location(location)
    # Tier 1 — exact (post-normalize).
    if norm in venue_lookup:
        return (venue_lookup[norm], "exact")
    # Tier 2 — fuzzy token_sort_ratio.
    best: int | None = None
    best_score: float = 0.0
    for vid, raw in venue_names_raw:
        score = fuzz.token_sort_ratio(norm, _normalize_location(raw))
        if score > best_score:
            best_score = score
            best = vid
    if best is not None and best_score >= threshold:
        return (best, f"fuzzy:{int(best_score)}")
    # Tier 3 — unresolved.
    return None


def assign_venue_id_if_null(
    session: Session,
    *,
    event_id: int,
    location: str,
    venue_lookup: dict[str, int],
    venue_names_raw: list[tuple[int, str]],
    threshold: int = _DEFAULT_THRESHOLD,
) -> tuple[int | None, str | None]:
    """CR-01-style guard for ``events.venue_id`` — set-once, never overwrite.

    Mirrors the referee idempotency block in
    ``src/ufc_prediction/scraper/ingest.py:470-474`` (Phase 22 CR-01 fix).
    Used by both the ``ufc scrape venues`` CLI and the integration tests
    in ``test_referee_persistence.py::TestIngestVenueIdempotency``.

    Returns:
        ``(venue_id, match_kind)``. The ``match_kind`` is one of:
          * ``"exact"`` / ``"fuzzy:NN"`` — fresh assignment from a NULL
            ``venue_id``; the FK was just set.
          * ``"preserved"`` — event already had a non-NULL ``venue_id``;
            the existing FK is returned and was NOT overwritten.
          * ``None`` (with ``venue_id=None``) — no match >= threshold and
            either the event is missing or the FK stays unchanged.
    """
    event = session.get(Event, event_id)
    if event is None:
        return (None, None)
    # CR-01: already resolved? short-circuit — never overwrite.
    if event.venue_id is not None:
        return (event.venue_id, "preserved")
    result = match_venue(location, venue_lookup, venue_names_raw, threshold=threshold)
    if result is None:
        return (None, None)
    vid, kind = result
    event.venue_id = vid
    session.flush()
    return (vid, kind)
