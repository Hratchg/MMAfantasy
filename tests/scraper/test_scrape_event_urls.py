"""Phase 40 Plan 40-01 Task 3 (CORPUS-V25-02) — forward-compat golden-file
regression for the BFO scrape_event_urls URL generator.

Mirrors the v2.4 CONTRACT-V24-03 forward-compat lock pattern. The fixture
at ``tests/scraper/fixtures/event_urls_2024.json`` pins the canonical
URL output of ``ufc_prediction.scraper.bfo_scraper.canonicalize_event_url``
for a frozen list of 2024-2026 sample events plus a small set of negative
drift cases (the WR-01 trailing-dash slug, non-/events paths, malformed
hrefs).

The fix point (canonicalize_event_url) is the new pure helper extracted
from ``_extract_event_name_and_url``; it consumes a relative BFO href
plus a fallback URL and returns the absolute canonical URL ONLY when the
href matches the tightened ``_EVENT_HREF_RE`` (which forbids trailing-
dash slugs such as ``foo--123``). On regex mismatch it returns the
fallback unchanged.

If this test fails in a future phase, the URL generator has drifted —
fix the generator OR (if the drift is intentional) regenerate the fixture
AND open a phase-scoped review of consumers (``scripts/bfo_backfill.py``,
the Plan 40-02 incremental-scrape entry, ``BFOScraper.scrape_event_urls``,
etc.).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "event_urls_2024.json"
)


@pytest.fixture(scope="module")
def fixture() -> dict:
    """Frozen event-URL golden file (CORPUS-V25-02 forward-compat lock)."""
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_fixture_exists_and_has_required_keys(fixture: dict) -> None:
    """Schema gate — the fixture must carry events + drift_negative_cases."""
    assert "events" in fixture, fixture.keys()
    assert "drift_negative_cases" in fixture, fixture.keys()
    assert fixture["events"], "fixture.events must contain ≥1 record"
    assert (
        len(fixture["events"]) >= 5
    ), f"fixture.events must hold ≥5 records (got {len(fixture['events'])})"


def test_fixture_covers_2024_minimum(fixture: dict) -> None:
    """Year-coverage gate — 2024 sample REQUIRED; 2025/2026 included when present."""
    years = {rec["event_date"][:4] for rec in fixture["events"]}
    assert "2024" in years, (
        f"fixture must include a 2024 event (got {sorted(years)!r})"
    )
    # 2025/2026 included opportunistically — sanity-check that if the
    # fixture claims those years, the dates are well-formed.
    for rec in fixture["events"]:
        yr = rec["event_date"][:4]
        assert yr.isdigit() and len(yr) == 4, rec["event_date"]


def test_scrape_event_urls_matches_golden_fixture(fixture: dict) -> None:
    """Per-record byte-compare of canonicalize_event_url output.

    The post-Phase-40 generator MUST return EXACTLY the fixture's
    ``expected_url`` for each record. ANY drift fails this assertion
    with a per-record diff (event_id + name + date) so the operator
    can localize which record drifted.
    """
    from ufc_prediction.scraper.bfo_scraper import canonicalize_event_url

    drift = []
    for record in fixture["events"]:
        href = record["bfo_href"]
        # Use an obviously-distinct fallback so any silent fallback hit
        # produces a screaming-loud mismatch instead of a false positive.
        fallback = "https://www.bestfightodds.com/__golden_fallback__"
        actual = canonicalize_event_url(href, fallback=fallback)
        expected = record["expected_url"]
        if actual != expected:
            drift.append(
                f"event_id={record['event_id']} "
                f"({record['event_name']!r} on {record['event_date']}): "
                f"expected={expected!r} actual={actual!r}"
            )

    assert not drift, (
        "scrape_event_urls URL generator drifted from frozen fixture:\n  - "
        + "\n  - ".join(drift)
    )


def test_drift_negative_cases_fall_back(fixture: dict) -> None:
    """Negative-case gate — drifted/malformed hrefs MUST fall back, not pass.

    Pins the WR-01 trailing-dash slug rejection (the original drift bug)
    plus the non-/events path and empty-href guards. If any of these
    starts SUCCESSFULLY producing a canonical URL in the future, the
    regex has been loosened — investigate before regenerating the
    fixture.
    """
    from ufc_prediction.scraper.bfo_scraper import canonicalize_event_url

    drift = []
    for record in fixture["drift_negative_cases"]:
        actual = canonicalize_event_url(
            record["bfo_href"], fallback=record["fallback"]
        )
        expected = record["expected_url"]
        if actual != expected:
            drift.append(
                f"{record['description']}: "
                f"href={record['bfo_href']!r} "
                f"expected={expected!r} actual={actual!r}"
            )

    assert not drift, (
        "Drift negative case(s) no longer fall back as expected — the "
        "URL canonicalizer regex may have been loosened:\n  - "
        + "\n  - ".join(drift)
    )


def test_event_href_regex_rejects_trailing_dash_slug() -> None:
    """Direct regex sanity test for the WR-01 lineage.

    Mirrors the regression assertion in
    ``tests/scripts/test_bfo_backfill.py`` so the WR-01 fix has a copy
    on the bfo_scraper side too.
    """
    from ufc_prediction.scraper.bfo_scraper import _EVENT_HREF_RE

    # Trailing-dash slug must NOT match (WR-01 bug class).
    assert _EVENT_HREF_RE.match("/events/foo--123") is None
    assert _EVENT_HREF_RE.match("/events/ufc-300-pereira-vs-hill--9244") is None
    # Well-formed slugs MUST match.
    assert _EVENT_HREF_RE.match("/events/ufc-300-pereira-vs-hill-9244") is not None
    assert _EVENT_HREF_RE.match("/events/a-1") is not None
