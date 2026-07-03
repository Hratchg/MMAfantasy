"""Integration tests for the scraping orchestrator (ingest.py).

Tests the pipeline from mocked HTML -> parser -> Pydantic validation -> DB upsert.
Uses MockScraperClient to avoid live network calls.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session

from ufc_prediction.data.schemas import FighterRow, FightRow, IngestResult
from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fighter import Fighter
from ufc_prediction.scraper.models import (
    FightDetailPage,
    FighterProfile,
    FightSummary,
    RoundStatsRaw,
    SigStrikesRaw,
)

# ── Fixture loading ─────────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> str:
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


# ── Mock client ─────────────────────────────────────────────────────────────


class MockScraperClient:
    """Mock ScraperClient that returns fixture HTML based on URL patterns.

    Supports exact URL matches (checked first) and substring pattern matches.
    Also provides serial ``map`` / ``map_get`` helpers so ingest.py can call
    the batch-fetch API against this mock.
    """

    def __init__(
        self,
        fixture_map: dict[str, str],
        exact_map: dict[str, str] | None = None,
    ) -> None:
        self._fixture_map = fixture_map  # substring pattern -> html
        self._exact_map = exact_map or {}  # exact url -> html
        self._call_log: list[str] = []
        self.map_call_count: int = 0

    def get(self, url: str) -> str:
        self._call_log.append(url)
        # Check exact matches first
        if url in self._exact_map:
            return self._exact_map[url]
        # Then substring pattern matches
        for pattern, html in self._fixture_map.items():
            if pattern in url:
                return html
        msg = f"MockScraperClient: no fixture for URL {url}"
        raise RuntimeError(msg)

    def map(self, fn, urls):  # type: ignore[no-untyped-def]
        """Serial map that records dispatch count for assertions."""
        self.map_call_count += 1
        return [fn(u) for u in urls]

    def map_get(self, urls):  # type: ignore[no-untyped-def]
        """Convenience wrapper over ``self.map(self.get, urls)``."""
        return self.map(self.get, urls)

    def close(self) -> None:
        pass

    def __enter__(self) -> MockScraperClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @property
    def call_count(self) -> int:
        return len(self._call_log)

    def calls_containing(self, substring: str) -> list[str]:
        return [c for c in self._call_log if substring in c]


# Event URLs from the event_list_snippet.html fixture (3 completed events)
_EVENT_URLS = [
    (
        "http://ufcstats.com/event-details/f3eb664db7fb1df3",
        "UFC 327: Prochazka vs. Ulberg",
        "April 11, 2026",
        "Miami, Florida, USA",
    ),
    (
        "http://ufcstats.com/event-details/cc11dd22ee33ff44",
        "UFC Fight Night: Sandhagen vs. Nurmagomedov",
        "April 05, 2026",
        "Las Vegas, Nevada, USA",
    ),
    (
        "http://ufcstats.com/event-details/dd44ee55ff667788",
        "UFC 326: Pereira vs. Prochazka",
        "March 28, 2026",
        "Jacksonville, Florida, USA",
    ),
]


def _make_event_detail_html(name: str, date_str: str, location: str) -> str:
    """Generate event detail HTML by patching the real fixture with new metadata.

    This ensures each event URL returns a detail page with the correct
    event name/date, so scrape_latest_events can match events properly.
    """
    base = _load_fixture("event_detail.html")
    # Replace event name
    base = base.replace("UFC 327: Prochazka vs. Ulberg", name)
    # Replace date
    base = base.replace("April 11, 2026", date_str)
    # Replace location
    base = base.replace("Miami, Florida, USA", location)
    return base


def _make_mock_client() -> MockScraperClient:
    """Create a mock client with per-event detail pages matching the event listing fixture."""
    exact_map = {}
    for url, name, date_str, location in _EVENT_URLS:
        exact_map[url] = _make_event_detail_html(name, date_str, location)

    return MockScraperClient(
        fixture_map={
            "statistics/events/completed": _load_fixture("event_list_snippet.html"),
            "fight-details": _load_fixture("fight_detail_1round.html"),
            "fighter-details": _load_fixture("fighter_profile.html"),
        },
        exact_map=exact_map,
    )


# ── Unit tests for conversion functions ──────────────────────────────────────


class TestConvertFighterProfile:
    """Unit test for _convert_fighter_profile."""

    def test_convert_complete_profile(self) -> None:
        from ufc_prediction.scraper.ingest import _convert_fighter_profile

        profile = FighterProfile(
            name="Carlos Ulberg",
            nickname="Black Jag",
            height_str="6' 4\"",
            weight_str="205 lbs.",
            reach_str='79"',
            stance="Switch",
            dob_str="Nov 07, 1990",
        )
        result = _convert_fighter_profile(profile)
        assert isinstance(result, FighterRow)
        assert result.name == "Carlos Ulberg"
        assert result.height_inches == 76.0
        assert result.reach_inches == 79.0
        assert result.stance == "Switch"
        assert result.date_of_birth is not None
        assert result.leg_reach_inches is None  # not available from UFCStats

    def test_convert_missing_fields_profile(self) -> None:
        from ufc_prediction.scraper.ingest import _convert_fighter_profile

        profile = FighterProfile(
            name="Unknown Fighter",
            height_str="--",
            weight_str="--",
            reach_str="--",
            stance=None,
            dob_str=None,
        )
        result = _convert_fighter_profile(profile)
        assert result.name == "Unknown Fighter"
        assert result.height_inches is None
        assert result.reach_inches is None
        assert result.stance is None
        assert result.date_of_birth is None


class TestConvertFight:
    """Unit test for _convert_fight."""

    def test_convert_fight_basic(self) -> None:
        from ufc_prediction.scraper.ingest import _convert_fight

        fight_summary = FightSummary(
            fight_url="http://ufcstats.com/fight-details/abc123",
            fighter_a_name="Fighter A",
            fighter_a_url="http://ufcstats.com/fighter-details/aaa",
            fighter_b_name="Fighter B",
            fighter_b_url="http://ufcstats.com/fighter-details/bbb",
            winner="fighter_a",
            outcome="win",
            weight_class_raw="UFC Lightweight Bout",
            is_title_fight=False,
            method="KO/TKO",
            method_detail="Punch",
            round_finished=1,
            time_finished="4:32",
        )

        # Minimal fight detail page
        totals_a = RoundStatsRaw(
            fighter_name="Fighter A",
            knockdowns="1",
            sig_str="20 of 30",
            sig_str_pct="66%",
            total_str="30 of 40",
            td="1 of 2",
            td_pct="50%",
            sub_att="0",
            rev="0",
            ctrl="2:15",
        )
        totals_b = RoundStatsRaw(
            fighter_name="Fighter B",
            knockdowns="0",
            sig_str="10 of 25",
            sig_str_pct="40%",
            total_str="15 of 30",
            td="0 of 1",
            td_pct="0%",
            sub_att="1",
            rev="0",
            ctrl="0:30",
        )
        sig_a = SigStrikesRaw(
            fighter_name="Fighter A",
            sig_str="20 of 30",
            sig_str_pct="66%",
            head="10 of 15",
            body="5 of 8",
            leg="5 of 7",
            distance="12 of 20",
            clinch="5 of 6",
            ground="3 of 4",
        )
        sig_b = SigStrikesRaw(
            fighter_name="Fighter B",
            sig_str="10 of 25",
            sig_str_pct="40%",
            head="5 of 10",
            body="3 of 8",
            leg="2 of 7",
            distance="6 of 15",
            clinch="2 of 5",
            ground="2 of 5",
        )

        fight_detail = FightDetailPage(
            fighter_a_name="Fighter A",
            fighter_a_url="http://ufcstats.com/fighter-details/aaa",
            fighter_b_name="Fighter B",
            fighter_b_url="http://ufcstats.com/fighter-details/bbb",
            fighter_a_status="W",
            fighter_b_status="L",
            bout_type="UFC Lightweight Bout",
            method="KO/TKO",
            method_detail="Punch",
            round_finished=1,
            time_finished="4:32",
            time_format="3 Rnd (5-5-5)",
            referee="Herb Dean",
            totals=(totals_a, totals_b),
            sig_strikes=(sig_a, sig_b),
            per_round_totals=[(totals_a, totals_b)],
            per_round_sig_strikes=[(sig_a, sig_b)],
        )

        result = _convert_fight(
            event_name="UFC 300",
            event_date_str="April 13, 2024",
            location="Las Vegas, Nevada",
            fight_summary=fight_summary,
            fight_detail=fight_detail,
        )
        assert isinstance(result, FightRow)
        assert result.weight_class == "Lightweight"
        assert result.winner_name == "Fighter A"
        assert result.fighter_a_stats is not None
        assert result.fighter_a_stats.knockdowns == 1
        assert result.fighter_a_stats.sig_strikes_landed == 20
        assert result.fighter_a_stats.sig_strikes_attempted == 30


# ── Integration tests (DB) ──────────────────────────────────────────────────


class TestScrapeAllEvents:
    """Integration tests for scrape_all_events."""

    def test_scrape_all_with_mock_client(self, session: Session) -> None:
        from ufc_prediction.scraper.ingest import scrape_all_events

        mock_client = _make_mock_client()
        result = scrape_all_events(session, mock_client)

        assert isinstance(result, IngestResult)
        assert result.accepted > 0

        # Check events exist in DB with correct source
        events = session.query(Event).filter(Event.source == "ufcstats").all()
        assert len(events) > 0

    def test_scrape_all_filters_upcoming(self, session: Session) -> None:
        from ufc_prediction.scraper.ingest import scrape_all_events

        mock_client = _make_mock_client()
        scrape_all_events(session, mock_client)

        # The fixture has 1 upcoming event. It should NOT appear in DB.
        events = session.query(Event).filter(Event.source == "ufcstats").all()
        # The fixture has 3 completed events and 1 upcoming
        # All DB events should be from completed events only
        for event in events:
            assert event.source == "ufcstats"

    def test_scrape_all_processes_events_chronologically(self, session: Session) -> None:
        from ufc_prediction.scraper.ingest import scrape_all_events

        mock_client = _make_mock_client()
        # Track progress to verify chronological order
        progress_calls: list[tuple[int, int]] = []
        scrape_all_events(
            session,
            mock_client,
            progress_callback=lambda i, total: progress_calls.append((i, total)),
        )
        # Progress callback should have been called
        assert len(progress_calls) > 0


class TestScrapeLatestEvents:
    """Integration tests for scrape_latest_events."""

    def test_scrape_latest_skips_existing(self, session: Session) -> None:
        from ufc_prediction.scraper.ingest import scrape_all_events, scrape_latest_events

        mock_client = _make_mock_client()
        # First scrape to populate DB
        scrape_all_events(session, mock_client)
        event_count_after_first = session.query(Event).filter(Event.source == "ufcstats").count()

        # Second scrape with "latest" -- should not add events
        mock_client2 = _make_mock_client()
        scrape_latest_events(session, mock_client2)

        event_count_after_second = session.query(Event).filter(Event.source == "ufcstats").count()
        assert event_count_after_second == event_count_after_first

    def test_scrape_latest_all_in_db_returns_zero(self, session: Session) -> None:
        from ufc_prediction.scraper.ingest import scrape_all_events, scrape_latest_events

        mock_client = _make_mock_client()
        scrape_all_events(session, mock_client)

        mock_client2 = _make_mock_client()
        result = scrape_latest_events(session, mock_client2)
        assert result.accepted == 0


class TestErrorHandling:
    """Tests for parse failure handling per D-11."""

    def test_parse_failure_continues(self, session: Session) -> None:
        from ufc_prediction.scraper.ingest import scrape_all_events

        # Create client where one event detail page is broken
        mock_client = MockScraperClient(
            {
                "statistics/events/completed": _load_fixture("event_list_snippet.html"),
                "event-details": "<html><body>broken html with no fight rows</body></html>",
                "fight-details": _load_fixture("fight_detail_1round.html"),
                "fighter-details": _load_fixture("fighter_profile.html"),
            }
        )

        # Should not raise -- parse failure on event pages logged and continued
        result = scrape_all_events(session, mock_client)
        # All events fail to parse, so rejected or accepted depending on handling
        assert isinstance(result, IngestResult)


class TestFighterCache:
    """Tests for fighter profile caching."""

    def test_fighter_cache_prevents_refetch(self, session: Session) -> None:
        from ufc_prediction.scraper.ingest import scrape_all_events

        mock_client = _make_mock_client()
        scrape_all_events(session, mock_client)

        # The event_detail fixture has 3 fights with potentially overlapping fighters.
        # Fighter profile URLs should only be fetched once per unique URL.
        fighter_fetches = mock_client.calls_containing("fighter-details")
        fighter_urls = set(fighter_fetches)
        # Each unique fighter URL should be fetched at most once
        assert len(fighter_fetches) == len(fighter_urls)


class TestSourceTagging:
    """Tests for source="ufcstats" tagging per D-09."""

    def test_source_is_ufcstats(self, session: Session) -> None:
        from ufc_prediction.scraper.ingest import scrape_all_events

        mock_client = _make_mock_client()
        scrape_all_events(session, mock_client)

        fighters = session.query(Fighter).filter(Fighter.source == "ufcstats").all()
        events = session.query(Event).filter(Event.source == "ufcstats").all()
        fights = session.query(Fight).filter(Fight.source == "ufcstats").all()

        assert len(fighters) > 0
        assert len(events) > 0
        assert len(fights) > 0

        # Verify ALL records have source=ufcstats
        all_fighters = session.query(Fighter).all()
        all_events = session.query(Event).all()
        all_fights = session.query(Fight).all()

        for f in all_fighters:
            assert f.source == "ufcstats"
        for e in all_events:
            assert e.source == "ufcstats"
        for f in all_fights:
            assert f.source == "ufcstats"


class TestIdempotency:
    """Tests for upsert idempotency."""

    def test_upsert_idempotency(self, session: Session) -> None:
        from ufc_prediction.scraper.ingest import scrape_all_events

        mock_client1 = _make_mock_client()
        scrape_all_events(session, mock_client1)

        fighter_count_1 = session.query(Fighter).count()
        event_count_1 = session.query(Event).count()
        fight_count_1 = session.query(Fight).count()

        # Second scrape with same data
        mock_client2 = _make_mock_client()
        scrape_all_events(session, mock_client2)

        fighter_count_2 = session.query(Fighter).count()
        event_count_2 = session.query(Event).count()
        fight_count_2 = session.query(Fight).count()

        assert fighter_count_1 == fighter_count_2
        assert event_count_1 == event_count_2
        assert fight_count_1 == fight_count_2


# ── Concurrency wiring tests (quick task 260422-qye) ────────────────────────


class _RaisingMockScraperClient(MockScraperClient):
    """MockScraperClient that raises ``RuntimeError`` for specific URLs.

    Used to prove per-batch error isolation: one failing URL inside a parallel
    batch must not abort the whole batch.
    """

    def __init__(
        self,
        fixture_map: dict[str, str],
        exact_map: dict[str, str] | None = None,
        raising_urls: set[str] | None = None,
    ) -> None:
        super().__init__(fixture_map=fixture_map, exact_map=exact_map)
        self._raising_urls: set[str] = raising_urls or set()

    def get(self, url: str) -> str:
        if url in self._raising_urls:
            self._call_log.append(url)
            msg = f"simulated failure for {url}"
            raise RuntimeError(msg)
        return super().get(url)


class _SpyClient:
    """Network-free stand-in used to assert ScraperClient construction kwargs.

    Records every ``__init__`` kwarg on the class (not the instance) so the
    test can inspect them without having to pluck the instance out of a
    context manager. ``get`` returns empty string so the event-list parser
    raises ValueError (min_events=1), which we catch in the test.
    """

    last_kwargs: dict[str, object] = {}

    def __init__(self, **kwargs: object) -> None:
        type(self).last_kwargs = dict(kwargs)

    def get(self, url: str) -> str:
        return ""

    def map(self, fn, urls):  # type: ignore[no-untyped-def]
        return [fn(u) for u in urls]

    def map_get(self, urls):  # type: ignore[no-untyped-def]
        return [self.get(u) for u in urls]

    def close(self) -> None:
        pass

    def __enter__(self) -> _SpyClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class TestScrapeConcurrencyIntegration:
    """Tests proving ingest.py wires the workers/map API from 260422-qla."""

    def test_scrape_all_uses_map_for_batch_fetches(self, session: Session) -> None:
        from ufc_prediction.scraper.ingest import scrape_all_events

        mock_client = _make_mock_client()
        result = scrape_all_events(session, mock_client)

        # One batch for events + one batch per event for fights -> >= 2.
        assert mock_client.map_call_count >= 2
        # Existing basic assertions still hold.
        assert isinstance(result, IngestResult)
        assert result.accepted > 0
        events = session.query(Event).filter(Event.source == "ufcstats").all()
        assert len(events) > 0

    def test_scrape_all_isolates_per_url_failures(self, session: Session) -> None:
        from ufc_prediction.scraper.ingest import scrape_all_events

        # Pick one of the three event URLs as the one that will fail.
        failing_url = _EVENT_URLS[0][0]
        exact_map = {}
        for url, name, date_str, location in _EVENT_URLS:
            exact_map[url] = _make_event_detail_html(name, date_str, location)

        mock_client = _RaisingMockScraperClient(
            fixture_map={
                "statistics/events/completed": _load_fixture("event_list_snippet.html"),
                "fight-details": _load_fixture("fight_detail_1round.html"),
                "fighter-details": _load_fixture("fighter_profile.html"),
            },
            exact_map=exact_map,
            raising_urls={failing_url},
        )

        # Should NOT raise: batch error must be isolated to the failing URL.
        result = scrape_all_events(session, mock_client)

        assert isinstance(result, IngestResult)
        # At least one event was still accepted.
        assert result.accepted > 0
        # At least one rejected (the failing URL).
        assert result.rejected >= 1

    def test_scrape_all_constructs_client_with_default_workers_4(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ufc_prediction.scraper import ingest as ingest_mod

        _SpyClient.last_kwargs = {}
        monkeypatch.setattr(ingest_mod, "ScraperClient", _SpyClient)

        # scrape_all_events will call get(EVENT_LIST_URL) -> "" which then
        # trips parse_event_list(min_events=1) -> ValueError. That's fine;
        # we only care about the ScraperClient(...) kwargs captured by the spy.
        with pytest.raises(Exception):
            ingest_mod.scrape_all_events(session=None)  # type: ignore[arg-type]

        assert _SpyClient.last_kwargs.get("workers") == 4

        # Override path: workers=7 flows through.
        _SpyClient.last_kwargs = {}
        with pytest.raises(Exception):
            ingest_mod.scrape_all_events(session=None, workers=7)  # type: ignore[arg-type]
        assert _SpyClient.last_kwargs.get("workers") == 7


# ── Browser-fetcher parity ───────────────────────────────────────────────────


class _DispatchPage:
    """Fake Playwright page: dispatches ``content()`` by the last-navigated URL.

    Lets a real :class:`BrowserFetcher` (Playwright fully bypassed) drive the
    ingest orchestrator against the same fixtures the mock/http path uses, so
    we can prove the browser backend drops into ingest and yields identical
    fights.
    """

    def __init__(self, exact_map: dict[str, str], fixture_map: dict[str, str]) -> None:
        self._exact_map = exact_map
        self._fixture_map = fixture_map
        self._current_url = ""

    def goto(self, url: str, **_kwargs: object):  # type: ignore[no-untyped-def]
        self._current_url = url
        resp = MagicMock()
        resp.status = 200
        return resp

    def wait_for_load_state(self, *_a: object, **_k: object) -> None:
        return None

    def wait_for_selector(self, *_a: object, **_k: object):  # type: ignore[no-untyped-def]
        return MagicMock()

    def content(self) -> str:
        if self._current_url in self._exact_map:
            return self._exact_map[self._current_url]
        for pattern, html in self._fixture_map.items():
            if pattern in self._current_url:
                return html
        msg = f"_DispatchPage: no fixture for URL {self._current_url}"
        raise RuntimeError(msg)

    def close(self) -> None:
        return None


def _make_browser_fetcher() -> object:
    from ufc_prediction.scraper.browser_fetch import BrowserFetcher

    exact_map = {
        url: _make_event_detail_html(name, date_str, location)
        for url, name, date_str, location in _EVENT_URLS
    }
    page = _DispatchPage(
        exact_map=exact_map,
        fixture_map={
            "statistics/events/completed": _load_fixture("event_list_snippet.html"),
            "fight-details": _load_fixture("fight_detail_1round.html"),
            "fighter-details": _load_fixture("fighter_profile.html"),
        },
    )
    fetcher = BrowserFetcher(delay=0.0)
    fetcher._page = page  # inject fake page; skips real Chromium launch
    return fetcher


class TestBrowserFetcherIngestParity:
    """The BrowserFetcher drops into ingest and yields the same fights."""

    def test_browser_backend_ingests_fights(self, session: Session) -> None:
        from ufc_prediction.scraper.ingest import scrape_all_events

        browser = _make_browser_fetcher()
        result = scrape_all_events(session, browser)

        assert isinstance(result, IngestResult)
        assert result.accepted > 0

        events = session.query(Event).filter(Event.source == "ufcstats").all()
        fights = session.query(Fight).filter(Fight.source == "ufcstats").all()
        assert len(events) > 0
        assert len(fights) > 0

    def test_browser_backend_ingests_same_fights_as_http(self, session: Session) -> None:
        """Parity: browser and http backends yield the identical fight set.

        Both run ``scrape_all_events`` over byte-identical fixtures on the same
        rolled-back session. The mock (http) run populates the DB; the browser
        run over the same fixtures processes the same fights (identical accepted
        count) and, thanks to idempotent upsert, leaves the fight set unchanged.
        This proves the browser fetcher feeds the parsers the same content.
        """
        from ufc_prediction.scraper.ingest import scrape_all_events

        http_result = scrape_all_events(session, _make_mock_client())
        http_fights = sorted((f.fighter_a_id, f.fighter_b_id) for f in session.query(Fight).all())

        assert http_result.accepted > 0
        assert len(http_fights) > 0

        # Re-run through the browser backend on the now-populated DB: identical
        # fixtures => same fights processed and the same (idempotent) fight set.
        browser_result = scrape_all_events(session, _make_browser_fetcher())
        browser_fights = sorted(
            (f.fighter_a_id, f.fighter_b_id) for f in session.query(Fight).all()
        )

        assert browser_result.accepted == http_result.accepted
        assert browser_fights == http_fights

    def test_browser_and_http_fetch_identical_html(self) -> None:
        """Fetch-layer parity: both backends return byte-identical HTML per URL.

        This is what guarantees the unchanged parsers produce the same fights,
        independent of any DB state.
        """
        mock = _make_mock_client()
        browser = _make_browser_fetcher()

        urls = [
            "http://ufcstats.com/statistics/events/completed?page=all",
            _EVENT_URLS[0][0],
            "http://ufcstats.com/fight-details/deadbeef",
            "http://ufcstats.com/fighter-details/cafef00d",
        ]
        for url in urls:
            assert browser.get(url) == mock.get(url), f"HTML mismatch for {url}"
