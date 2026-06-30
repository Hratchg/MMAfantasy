"""Phase 28 — INGEST-V23-02 venue_match three-tier matcher unit tests.

Covers:
- Tier 1 exact string match (post-normalize) -> (venue_id, "exact")
- Tier 2 rapidfuzz.fuzz.token_sort_ratio >= threshold -> (venue_id, "fuzzy:NN")
- Tier 3 no match >= threshold -> None (UNRESOLVED; CLI surfaces to operator)
- NFKD accent fold + lowercase + whitespace collapse normalization
- token_sort_ratio (NOT fuzz.ratio) is the chosen scorer per CONTEXT D-06

Mirrors the bfo_matcher.py test discipline (Phase 15) but extends to the
location-string domain (word-order variation; e.g. "Las Vegas, USA, Nevada"
vs "Las Vegas, Nevada, USA") which fuzz.ratio handles poorly.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ufc_prediction.scraper.venue_match import (
    _normalize_location,
    match_venue,
)


def _load_venues_lookup() -> tuple[dict[str, int], list[tuple[int, str]]]:
    """Load data/venues.csv post-Plan-28-01 and build the two lookup structures.

    Returns (venue_lookup, venue_names_raw) where:
      - venue_lookup: normalized name -> venue_id
      - venue_names_raw: list of (venue_id, raw_name) tuples for fuzzy fallback
    """
    venues_path = Path(__file__).resolve().parents[2] / "data" / "venues.csv"
    venue_lookup: dict[str, int] = {}
    venue_names_raw: list[tuple[int, str]] = []
    with venues_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = int(row["venue_id"])
            raw = row["name"]
            venue_lookup[_normalize_location(raw)] = vid
            venue_names_raw.append((vid, raw))
    return venue_lookup, venue_names_raw


@pytest.fixture(scope="module")
def venue_lookup_fixture() -> dict[str, int]:
    lookup, _ = _load_venues_lookup()
    return lookup


@pytest.fixture(scope="module")
def venue_names_raw_fixture() -> list[tuple[int, str]]:
    _, names_raw = _load_venues_lookup()
    return names_raw


@pytest.fixture(scope="module")
def vegas_id(venue_lookup_fixture: dict[str, int]) -> int:
    """The venue_id for Las Vegas (the biggest single venue in the corpus)."""
    # post-Plan-28-01 row 53 is "Las Vegas, Nevada, USA" per 28-01 summary.
    vid = venue_lookup_fixture.get(_normalize_location("Las Vegas, Nevada, USA"))
    assert vid is not None, "Las Vegas not in venues.csv (Plan 28-01 substrate broken)"
    return vid


class TestNormalizeLocation:
    """Pure-function tests for _normalize_location."""

    def test_lowercase_fold(self) -> None:
        assert _normalize_location("LAS VEGAS, USA") == "las vegas, usa"

    def test_strips_accents(self) -> None:
        # São Paulo -> Sao Paulo via NFKD + ascii encode
        assert _normalize_location("São Paulo, Brasil") == "sao paulo, brasil"

    def test_collapses_whitespace(self) -> None:
        # Multiple internal spaces + leading/trailing whitespace
        assert _normalize_location("  Las  Vegas,   USA  ") == "las vegas, usa"

    def test_idempotent(self) -> None:
        once = _normalize_location("São Paulo,  Brasil")
        twice = _normalize_location(once)
        assert once == twice


class TestVenueMatchTiers:
    """Three-tier match: exact -> fuzzy>=85 -> None."""

    def test_exact_match_returns_venue_id_and_exact_kind(
        self,
        venue_lookup_fixture: dict[str, int],
        venue_names_raw_fixture: list[tuple[int, str]],
        vegas_id: int,
    ) -> None:
        result = match_venue(
            "Las Vegas, Nevada, USA",
            venue_lookup_fixture,
            venue_names_raw_fixture,
        )
        assert result == (vegas_id, "exact")

    def test_fuzzy_match_above_85_returns_venue_id_and_fuzzy_kind(
        self,
        venue_lookup_fixture: dict[str, int],
        venue_names_raw_fixture: list[tuple[int, str]],
        vegas_id: int,
    ) -> None:
        # Word-order variation that fuzz.ratio handles poorly but
        # fuzz.token_sort_ratio handles perfectly (score ~100).
        result = match_venue(
            "Las Vegas, USA, Nevada",
            venue_lookup_fixture,
            venue_names_raw_fixture,
        )
        assert result is not None
        vid, kind = result
        assert vid == vegas_id
        assert kind.startswith("fuzzy:")
        score = int(kind.split(":", 1)[1])
        assert score >= 85

    def test_fuzzy_match_below_85_returns_none(self) -> None:
        # Vegas-only lookup; query a completely unrelated location.
        tiny_lookup = {"las vegas, nevada, usa": 1}
        tiny_names = [(1, "Las Vegas, Nevada, USA")]
        # token_sort_ratio between completely different strings is <85.
        result = match_venue("Tokyo, Japan", tiny_lookup, tiny_names)
        assert result is None

    def test_nfkd_normalization_strips_accents(self) -> None:
        # Construct a tiny lookup using the non-accent form; the input has
        # accents — after NFKD fold the exact match should hit.
        lookup = {"sao paulo, brasil": 42}
        names_raw = [(42, "Sao Paulo, Brasil")]
        result = match_venue("São Paulo, Brasil", lookup, names_raw)
        assert result == (42, "exact")

    def test_whitespace_collapse_yields_exact_match(self) -> None:
        lookup = {"las vegas, usa": 7}
        names_raw = [(7, "Las Vegas, USA")]
        result = match_venue("  Las  Vegas,   USA  ", lookup, names_raw)
        assert result == (7, "exact")

    def test_token_sort_ratio_used_not_ratio(self) -> None:
        # "London, England, UK" vs "UK, England, London":
        #   fuzz.ratio       — ~10-30 (chars don't line up at all)
        #   fuzz.token_sort_ratio — 100 (same tokens, sorted match)
        # If venue_match uses token_sort_ratio (per CONTEXT D-06) this returns
        # the venue with "fuzzy:100". If it (incorrectly) used fuzz.ratio
        # the score would be far below 85 and we'd get None.
        lookup = {"london, england, uk": 99}
        names_raw = [(99, "London, England, UK")]
        result = match_venue("UK, England, London", lookup, names_raw)
        assert result is not None, (
            "token_sort_ratio should score this 100; if None, fuzz.ratio "
            "is being used (CONTEXT D-06 binding violated)"
        )
        vid, kind = result
        assert vid == 99
        # Word-order rearrangement is NOT exact (normalization preserves
        # comma/space positions), so it must be the fuzzy tier.
        assert kind.startswith("fuzzy:")
        score = int(kind.split(":", 1)[1])
        # token_sort_ratio of "uk england london" vs "london england uk" is
        # ~95 (commas inside the normalized string shift in the sort);
        # fuzz.ratio of the same pair is ~58. Anything in the >=85 range
        # is concrete proof token_sort_ratio (not fuzz.ratio) is the scorer.
        assert score >= 85
