"""Tests for the in-house BFO scraper (Plan 15-01).

Wave 0 stubs: while ``ufc_prediction.scraper.bfo_scraper`` does not yet
exist, ``pytest.importorskip`` keeps the suite green. Once Task 2 lands
the module the tests turn green / red according to actual behavior.

Coverage:
- parser tests on captured BFO fixtures (table.team-stats-table)
- blank-moneyline -> None handling
- malformed-HTML / CAPTCHA -> typed BFOParseError
- search-result fuzzy match (Jon Jones)
- CSV column order matches BFOOddsIngester input format
- ``client.map`` (thread pool) is invoked from ``scrape_all``
- module never imports ``multiprocessing`` (the literal ODDS-01 fix)
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def bfo():
    """Import the bfo_scraper module lazily.

    Wave 0: while ``ufc_prediction.scraper.bfo_scraper`` does not yet
    exist, ``pytest.importorskip`` keeps each test green-skipped. Once
    Plan 15-01 Task 2 ships the module the fixture returns it for real.
    """
    return pytest.importorskip("ufc_prediction.scraper.bfo_scraper")


# ── Parser tests ────────────────────────────────────────────────────────────


def test_parse_fighter_page_extracts_opening_closing(bfo) -> None:
    """Real BFO fighter HTML yields fights with integer opening/closing odds."""
    html = (FIXTURES / "bfo_fighter.html").read_text(encoding="utf-8")
    page = bfo.parse_bfo_fighter_page(html, "https://www.bestfightodds.com/fighters/Jon-Jones-819")
    assert page.name, "fighter name must be non-empty on a real BFO profile page"
    assert len(page.fights) > 0, "Jon Jones page must yield at least one fight"
    # At least one historical fight must have all three integer moneylines
    has_integer_odds = any(
        isinstance(f.opening, int)
        and isinstance(f.closing_range_min, int)
        and isinstance(f.closing_range_max, int)
        for f in page.fights
    )
    assert has_integer_odds, "expected at least one fight with full integer odds"


def test_parse_fighter_page_handles_blank_moneylines(bfo) -> None:
    """Blank ``<td.moneyline>`` cells map to ``None`` rather than 0."""
    html = (FIXTURES / "bfo_fighter_no_odds.html").read_text(encoding="utf-8")
    page = bfo.parse_bfo_fighter_page(
        html, "https://www.bestfightodds.com/fighters/Test-Fighter-1234"
    )
    # Must extract at least one fight where all moneylines are None
    none_only = [
        f
        for f in page.fights
        if f.opening is None and f.closing_range_min is None and f.closing_range_max is None
    ]
    assert len(none_only) >= 1, (
        "expected at least one blank-moneyline fight to be emitted with None odds"
    )


def test_parse_fighter_page_raises_on_missing_table(bfo) -> None:
    """When the team-stats-table selector misses, raise BFOParseError (T-15-01-01)."""
    with pytest.raises(bfo.BFOParseError):
        bfo.parse_bfo_fighter_page(
            "<html><body>nope</body></html>",
            "https://www.bestfightodds.com/fighters/Nobody-0",
        )


def test_parse_fighter_page_detects_captcha(bfo) -> None:
    """An ``id="hfmr8"`` element must trigger a CAPTCHA-flavored BFOParseError."""
    captcha_html = (
        "<html><body>"
        '<div id="hfmr8">Verify you are human by completing the action below.</div>'
        "</body></html>"
    )
    with pytest.raises(bfo.BFOParseError) as excinfo:
        bfo.parse_bfo_fighter_page(captcha_html, "https://www.bestfightodds.com/fighters/Anyone")
    # Disposition T-15-01-02 — caller must be able to detect captcha vs structural
    assert "captcha" in str(excinfo.value).lower()


# ── Search / fuzzy-match tests ──────────────────────────────────────────────


def test_search_match_returns_best_fuzz(bfo) -> None:
    """The Jon Jones search page resolves to a /fighters/ URL."""
    html = (FIXTURES / "bfo_search.html").read_text(encoding="utf-8")
    url = bfo.find_bfo_fighter_url("Jon Jones", html, threshold=80)
    assert url is not None, "Jon Jones must match SOMETHING on the BFO search page"
    assert url.startswith("https://www.bestfightodds.com/fighters/"), url


def test_search_match_returns_none_below_threshold(bfo) -> None:
    """A garbage name well below the fuzz threshold returns None."""
    html = (FIXTURES / "bfo_search.html").read_text(encoding="utf-8")
    url = bfo.find_bfo_fighter_url("Totally Different Name 9999", html, threshold=80)
    assert url is None


# ── CSV / scrape_all tests ──────────────────────────────────────────────────


def test_writes_correct_csv_format(bfo, tmp_path: Path) -> None:
    """``_write_odds_csv`` emits the exact column order BFOOddsIngester reads.

    Blank-cell convention for ``None`` mirrors ``BFOOddsRow._coerce_blank_ml``.
    """
    from ufc_prediction.scraper.bfo_models import BFOOddsRow

    scraper = bfo.BFOScraper(
        client=MagicMock(),
        session=MagicMock(),
        data_folder=tmp_path,
        workers=1,
    )

    rows = [
        BFOOddsRow(
            fight_id="2023-03-04|819|10463",
            fighter_id="819",
            opening=140,
            closing_range_min=-186,
            closing_range_max=-143,
        ),
        BFOOddsRow(
            fight_id="upcoming|819|99",
            fighter_id="819",
            opening=None,
            closing_range_min=None,
            closing_range_max=None,
        ),
    ]
    scraper._write_odds_csv(rows)

    out = tmp_path / "BestFightOdds_odds.csv"
    assert out.exists()
    with out.open(newline="") as fh:
        reader = list(csv.reader(fh))
    assert reader[0] == [
        "fight_id",
        "fighter_id",
        "opening",
        "closing_range_min",
        "closing_range_max",
    ]
    # First row: integer cells stringified
    assert reader[1] == [
        "2023-03-04|819|10463",
        "819",
        "140",
        "-186",
        "-143",
    ]
    # Second row: blank cells for None (NOT "None" or "0")
    assert reader[2] == ["upcoming|819|99", "819", "", "", ""]


def test_uses_thread_pool(bfo) -> None:
    """``scrape_all`` invokes ``client.map`` (thread pool) instead of multiprocessing."""

    client = MagicMock()
    # Two map calls: search stage (returns search HTMLs), profile stage (returns
    # profile HTMLs). Returning empty results short-circuits the parse loop and
    # keeps the test pure-unit.
    client.map.return_value = []

    session = MagicMock()
    # Make the DB fetch return a single fighter so search_urls is non-empty
    # but the loop still terminates quickly (we mock map to return []).
    exec_result = MagicMock()
    exec_result.all.return_value = [(1, "Jon Jones")]
    session.execute.return_value = exec_result

    scraper = bfo.BFOScraper(
        client=client,
        session=session,
        data_folder=Path("/tmp/bfo_test_does_not_need_to_exist"),
        workers=4,
    )
    # We call scrape_all but don't assert on its return — only on map usage.
    # If scrape_all branches to multiprocessing, client.map would not be called.
    try:  # noqa: SIM105
        scraper.scrape_all()
    except Exception:
        # Errors downstream of `client.map` are fine — we only care that map was hit.
        pass

    assert client.map.called, "BFOScraper.scrape_all must invoke client.map (thread pool)"
    # The first positional arg is a callable (functools.partial wrapping the fetcher)
    first_call_args, _kwargs = client.map.call_args_list[0]
    assert callable(first_call_args[0]), "client.map must be called with a callable"


def test_no_multiprocessing() -> None:
    """The bfo_scraper module must not import ``multiprocessing`` (ODDS-01).

    This is the literal source-level fix for the PicklingError class:
    ``ScraperClient`` uses a ``ThreadPoolExecutor`` instead, and any
    accidental reintroduction of multiprocessing in future edits would
    fail this test.
    """
    src_path = Path("src/ufc_prediction/scraper/bfo_scraper.py")
    if not src_path.exists():
        pytest.skip("bfo_scraper.py not yet created (Wave 0)")
    src = src_path.read_text(encoding="utf-8")
    assert "import multiprocessing" not in src, "bfo_scraper must not import multiprocessing"
    assert "from multiprocessing" not in src, "bfo_scraper must not import from multiprocessing"


def test_canonical_fight_id_is_pair_symmetric(bfo) -> None:
    """``_canonical_fight_id`` must produce the SAME id for (A,B) and (B,A).

    Without this, scraping fighter A emits ``date|A|B`` and scraping fighter B
    emits ``date|B|A``; the ingester then sees 1 row per key instead of 2 and
    skips every fight as "expected 2 rows, got 1" (the bug that surfaced on
    the first live scrape: 21692 fights scanned → 0 rows upserted).
    """
    from datetime import date

    a_pov = bfo._canonical_fight_id(date(2021, 9, 17), "3227", "8697")
    b_pov = bfo._canonical_fight_id(date(2021, 9, 17), "8697", "3227")
    assert a_pov == b_pov, f"Pair symmetry broken: A-POV={a_pov!r} B-POV={b_pov!r}"
    # Lexicographic min-then-max (sorted as strings; numeric strings still
    # sort correctly when same length, and length is BFO-internal so we
    # tolerate either ordering as long as it's stable).
    assert "|3227|" in a_pov and "|8697" in a_pov


def test_scrape_all_drops_date_max_sentinel(bfo) -> None:
    """``scrape_all`` must drop ``event_date == date.max`` rows before CSV write.

    The parser legitimately emits ``date.max`` for upcoming fights with no
    committed event date (so the empty-moneyline test can verify all-None
    round-trip). But these rows have no resolved outcome and cannot be
    matched to a DB Fight, so they should be filtered before the CSV emit.

    Source-level check: scrape_all must contain an explicit ``date.max``
    drop guard. Future regressions that remove the guard would re-introduce
    the 9999-12-31 leakage seen on the first live scrape.
    """
    src_path = Path("src/ufc_prediction/scraper/bfo_scraper.py")
    if not src_path.exists():
        pytest.skip("bfo_scraper.py not yet created (Wave 0)")
    src = src_path.read_text(encoding="utf-8")
    assert "date.max" in src, "scrape_all must reference date.max to drop sentinel rows"
    # Specifically: somewhere in the file, there's a guard like
    # `if fight.event_date == date.max: continue` (or equivalent)
    assert "== date.max" in src, (
        "scrape_all must explicitly drop fights with event_date == date.max"
    )


# ── Phase 21 Plan 21-01 Task 3: scrape_event_urls (URL-list mode) ───────────


class TestScrapeEventUrls:
    """Phase 21 Task 3 — extend BFOScraper with ``scrape_event_urls``.

    URL-list scrape mode (NEW) — the inverse of ``scrape_all``: caller
    provides a curated list of BFO event URLs (from the Phase 21 backfill
    driver's gap query), the method fetches each event page, parses
    per-matchup odds (range across bookies), and writes a CSV in the
    EXACT ufcscraper-compatible column order so the existing
    ``BFOOddsIngester._load_odds_rows`` reads it without modification.

    The composite ``fight_id`` format ``"{event.date_iso}|{a_slug}|{b_slug}"``
    is preserved (999.4 / Couture-Liddell rematch fix).
    """

    def _make_scraper(self, bfo, tmp_path: Path, html_for_url: dict[str, str]):
        """Build a BFOScraper whose mocked client.get returns per-URL HTML.

        ``html_for_url`` maps URL string -> HTML body. Any URL not in
        the dict raises a RuntimeError to surface unexpected requests.
        """
        from unittest.mock import MagicMock

        client = MagicMock()

        def _get(url: str) -> str:
            if url not in html_for_url:
                raise RuntimeError(f"Unexpected URL fetched: {url}")
            return html_for_url[url]

        client.get.side_effect = _get
        scraper = bfo.BFOScraper(
            client=client,
            session=MagicMock(),
            data_folder=tmp_path,
            workers=1,
        )
        return scraper, client

    def test_scrape_event_urls_writes_csv_in_ufcscraper_format(
        self,
        bfo,
        tmp_path: Path,
    ) -> None:
        """One event URL → CSV with 5 columns + ≥2 rows per matchup."""
        event_html = (FIXTURES / "bfo_event.html").read_text(encoding="utf-8")
        url = "https://www.bestfightodds.com/events/cage-warriors-205-4161"
        scraper, _client = self._make_scraper(bfo, tmp_path, {url: event_html})

        out_path = tmp_path / "url_list_odds.csv"
        result = scraper.scrape_event_urls([url], out_path)

        assert result == out_path
        assert out_path.exists()
        with out_path.open(newline="") as fh:
            reader = list(csv.reader(fh))
        # Header in BFOOddsIngester column order (D-04 of Phase 15).
        assert reader[0] == [
            "fight_id",
            "fighter_id",
            "opening",
            "closing_range_min",
            "closing_range_max",
        ]
        # At least one matchup parsed → ≥2 fighter rows.
        assert len(reader) >= 3, f"Expected header + ≥2 fighter rows; got {len(reader)} rows"

    def test_scrape_event_urls_composite_fight_id_parses_via_bfo_models(
        self,
        bfo,
        tmp_path: Path,
    ) -> None:
        """Each emitted ``fight_id`` round-trips via ``BFOOddsRow.event_date``.

        Ensures the composite ``"{date_iso}|{a_slug}|{b_slug}"`` format
        is preserved (the 999.4 regression guard — without the date prefix
        the rematch-overwrite bug returns).
        """
        from ufc_prediction.scraper.bfo_models import BFOOddsRow

        event_html = (FIXTURES / "bfo_event.html").read_text(encoding="utf-8")
        url = "https://www.bestfightodds.com/events/cage-warriors-205-4161"
        scraper, _client = self._make_scraper(bfo, tmp_path, {url: event_html})

        out_path = tmp_path / "url_list_odds.csv"
        scraper.scrape_event_urls([url], out_path)

        with out_path.open(newline="") as fh:
            reader = list(csv.DictReader(fh))
        assert reader, "scrape_event_urls produced no rows"
        # First data row must have a parseable composite fight_id.
        sample = BFOOddsRow(
            fight_id=reader[0]["fight_id"],
            fighter_id=reader[0]["fighter_id"],
            opening=reader[0]["opening"] or None,
            closing_range_min=reader[0]["closing_range_min"] or None,
            closing_range_max=reader[0]["closing_range_max"] or None,
        )
        # The event_date property MUST parse without raising — the 999.4
        # regression guard. The actual date value depends on the fixture
        # (Cage Warriors 205 is dated 2026-04-26 in the fixture's meta
        # description).
        from datetime import date

        assert isinstance(sample.event_date, date), (
            "Composite fight_id must round-trip through BFOOddsRow.event_date"
        )

    def test_scrape_event_urls_handles_captcha(
        self,
        bfo,
        tmp_path: Path,
    ) -> None:
        """Captcha-gated event page must NOT emit any rows for that URL.

        Existing ``_check_captcha`` semantics: an ``id="hfmr8"`` element
        signals Cloudflare gate; the scraper logs and skips that event,
        but other URLs in the batch continue.
        """
        captcha_html = '<html><body><div id="hfmr8">Verify you are human.</div></body></html>'
        url = "https://www.bestfightodds.com/events/some-captcha-event-9999"
        scraper, _client = self._make_scraper(bfo, tmp_path, {url: captcha_html})

        out_path = tmp_path / "captcha_only.csv"
        scraper.scrape_event_urls([url], out_path)

        # Output CSV exists with ONLY the header (no data rows).
        assert out_path.exists()
        with out_path.open(newline="") as fh:
            reader = list(csv.reader(fh))
        assert len(reader) == 1, f"Captcha URL must yield 0 data rows; got {len(reader) - 1}"

    def test_scrape_event_urls_skips_invalid_html(
        self,
        bfo,
        tmp_path: Path,
    ) -> None:
        """Homepage-fallback HTML (no event content) is logged + skipped."""
        from ufc_prediction.scraper.bfo_classify import BFO_HOMEPAGE_TITLE

        homepage_html = (
            f"<html><head><title>{BFO_HOMEPAGE_TITLE}</title></head>"
            "<body>UFC &amp; MMA Odds &amp; Betting Lines</body></html>"
        )
        url = "https://www.bestfightodds.com/events/nonexistent-99999"
        scraper, _client = self._make_scraper(bfo, tmp_path, {url: homepage_html})

        out_path = tmp_path / "homepage_skip.csv"
        scraper.scrape_event_urls([url], out_path)

        assert out_path.exists()
        with out_path.open(newline="") as fh:
            reader = list(csv.reader(fh))
        # Header only — no rows for the homepage-fallback URL.
        assert len(reader) == 1, f"Homepage-fallback URL must yield 0 rows; got {len(reader) - 1}"

    def test_scrape_event_urls_csv_consumable_by_ingester(
        self,
        bfo,
        tmp_path: Path,
    ) -> None:
        """Output CSV is loadable via ``BFOOddsIngester._load_odds_rows``.

        End-to-end smoke check: the CSV that ``scrape_event_urls`` writes
        must be readable by the EXISTING ingester without any code change
        (per Plan 21-01 acceptance criterion, anti-pattern #3 from
        21-PATTERNS.md: ``BFOOddsIngester`` is UNCHANGED).
        """
        from unittest.mock import MagicMock

        from ufc_prediction.scraper.bfo_ingest import BFOOddsIngester

        event_html = (FIXTURES / "bfo_event.html").read_text(encoding="utf-8")
        url = "https://www.bestfightodds.com/events/cage-warriors-205-4161"
        scraper, _client = self._make_scraper(bfo, tmp_path, {url: event_html})

        # Write CSV to the data_folder under the canonical ufcscraper name
        # so the ingester's _resolve_csv() finds it on the primary path.
        out_path = tmp_path / "BestFightOdds_odds.csv"
        scraper.scrape_event_urls([url], out_path)

        # Build an ingester pointed at the same folder; only invoke the
        # internal CSV-load step (avoid DB I/O — the data_folder + Pydantic
        # validation IS the smoke surface we need to verify).
        ingester = BFOOddsIngester(
            session=MagicMock(),
            data_folder=tmp_path,
            date_window_days=14,
        )
        odds_by_fight = ingester._load_odds_rows(out_path)
        assert odds_by_fight, (
            "Ingester _load_odds_rows produced 0 fights from scrape_event_urls output"
        )
        # Every loaded fight must have exactly 2 fighter rows (the
        # 2-rows-per-fight invariant the ingester enforces in ingest_all).
        for fight_id, rows in odds_by_fight.items():
            assert len(rows) == 2, (
                f"Fight {fight_id!r} has {len(rows)} rows "
                f"(expected 2 — matches ingest_all's invariant)"
            )

    def test_scrape_event_urls_csv_fighter_ids_resolve_through_match_fighters(
        self,
        bfo,
        tmp_path: Path,
    ) -> None:
        """REGRESSION TRIPWIRE (REVIEW.md CR-01) — slug-vs-numeric keyspace drift.

        The Plan 21-01 Task 3 source defect: ``scrape_event_urls`` emits
        ``fighter_id`` as the BFO URL slug (e.g. ``"Justin-Burlinson-7808"``),
        but ``BFOOddsIngester._match_fighters`` builds its ``bfo_to_db``
        lookup keyed by ``BFOFighterName.fighter_id`` which in production is
        the BFO **numeric** ID (the suffix of the slug, e.g. ``"7808"``).
        The two keyspaces never intersect, so ``ingest_all`` falls through
        ``bfo_to_db.get(row.fighter_id) is None → continue`` for every row —
        the EXACT vector that produced the original ``rows_upserted=0``
        incident (Apr 30, 2026).

        This test runs ``scrape_event_urls`` end-to-end, then drives the
        REAL ``BFOOddsIngester._match_fighters`` against a names-CSV that
        uses the production numeric-ID schema. It asserts that the
        ``odds_row.fighter_id`` values (slugs) and the ``bfo_to_db`` keys
        (numeric IDs from BFOFighterName) DO NOT intersect — documenting
        the defect at the source layer.

        Disposition: source-code fix is deferred to v2.3+ housekeeping
        (Backlog Item 10) because changing ``scrape_event_urls`` /
        ``parse_bfo_event_page`` emission format would require coordinated
        updates to the existing Plan 21-01 Task 3 slug-format test surface.
        Until then, production correctness depends on the data-layer
        workaround (commit 22018d6). When the source fix lands,
        ``slug_keyspace.isdisjoint(numeric_keyspace)`` will flip to False
        and this test will need to be flipped to a positive assertion
        (``assert resolved_count > 0``).
        """
        from unittest.mock import MagicMock

        from ufc_prediction.scraper.bfo_ingest import (
            BFOOddsIngester,
            IngestSummary,
        )
        from ufc_prediction.scraper.bfo_models import (
            BFOFighterName,
        )

        event_html = (FIXTURES / "bfo_event.html").read_text(encoding="utf-8")
        url = "https://www.bestfightodds.com/events/cage-warriors-205-4161"
        scraper, _client = self._make_scraper(bfo, tmp_path, {url: event_html})

        out_path = tmp_path / "BestFightOdds_odds.csv"
        scraper.scrape_event_urls([url], out_path)

        # Extract slug-format fighter_ids from the CSV (the keyspace the
        # ingester WILL look up via ``bfo_to_db.get(row.fighter_id)`` at
        # line ~146 of bfo_ingest.py).
        ingester = BFOOddsIngester(
            session=MagicMock(),
            data_folder=tmp_path,
            date_window_days=14,
        )
        odds_by_fight = ingester._load_odds_rows(out_path)
        assert odds_by_fight, "scrape_event_urls produced 0 fights"
        slug_keyspace: set[str] = set()
        for _fight_id, rows in odds_by_fight.items():
            for r in rows:
                slug_keyspace.add(r.fighter_id)
        # A real BFO event fixture has a slug like "Tom-Mearns-9510" —
        # contains a hyphen + ends in a numeric suffix. The numeric ID
        # alone would be just "9510" (no hyphens). Tripwire: at least one
        # emitted fighter_id MUST contain a hyphen (i.e., it IS a slug
        # not a numeric ID).
        assert any("-" in fid for fid in slug_keyspace), (
            "Source-defect tripwire fired: scrape_event_urls is no longer "
            "emitting slug-format fighter_ids. The v2.3+ backlog fix may "
            "have landed — flip this test to a positive assertion against "
            "_match_fighters and remove the deferred-to-v2.3 disposition "
            "comment."
        )

        # Build a synthetic ``bfo_names`` dict keyed by the PRODUCTION
        # numeric-ID schema (extract the numeric suffix from each slug).
        # This mirrors what fighters_names.csv would contain if it were
        # populated from a real BFO crawl (vs the header-only placeholder
        # the current --scrape mode writes).
        import re as _re

        _SUFFIX_RE = _re.compile(r"-(\d+)$")
        bfo_names: dict[str, list[BFOFighterName]] = {}
        for slug in slug_keyspace:
            m = _SUFFIX_RE.search(slug)
            if m is None:
                continue
            numeric_id = m.group(1)
            # Use a placeholder DB-matchable name (the actual name doesn't
            # matter for the keyspace assertion below).
            bfo_names[numeric_id] = [
                BFOFighterName(
                    fighter_id=numeric_id,
                    database="ufcstats",
                    name="Placeholder Name",
                    database_id=None,
                ),
            ]
        assert bfo_names, (
            "Could not extract any numeric IDs from slug keyspace — "
            "fixture is malformed for this regression test."
        )

        # Drive the REAL _match_fighters against the numeric-keyed names.
        # The session is mocked to return a single DB candidate so the
        # fuzzy-matcher has SOMETHING to match against (we don't care
        # whether it matches — we care about the KEYSPACE of bfo_to_db).
        session_mock = MagicMock()
        # ``_match_fighters`` calls
        # ``self._session.execute(select(Fighter.id, Fighter.name)).all()``
        # — return one candidate so the match attempt doesn't trivially
        # short-circuit.
        result_mock = MagicMock()
        result_mock.all.return_value = [SimpleNamespace(id=1, name="Placeholder Name")]
        session_mock.execute.return_value = result_mock
        ingester_with_session = BFOOddsIngester(
            session=session_mock,
            data_folder=tmp_path,
            date_window_days=14,
        )
        summary = IngestSummary()
        bfo_to_db = ingester_with_session._match_fighters(bfo_names, summary)

        # Keyspace assertion: the ingester's ``bfo_to_db`` is keyed by
        # numeric IDs (from bfo_names). The CSV's fighter_id column holds
        # slugs. They share NO common keys. This documents the defect.
        resolved_via_slug = sum(1 for slug in slug_keyspace if slug in bfo_to_db)
        assert resolved_via_slug == 0, (
            "CR-01 tripwire flipped: scrape_event_urls slug fighter_ids "
            "now resolve through _match_fighters. The v2.3+ source fix "
            "has landed — update this test to assert the new positive "
            "behavior."
        )
        # And conversely: bfo_to_db has numeric keys, none of which
        # appear in the CSV's slug keyspace. (This is the same defect
        # viewed from the other direction.)
        if bfo_to_db:
            assert slug_keyspace.isdisjoint(bfo_to_db.keys()), (
                "Unexpected partial overlap between slug keyspace and "
                "numeric keyspace — fixture or schema may have drifted."
            )
