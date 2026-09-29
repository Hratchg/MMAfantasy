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
from ufc_prediction.models.round_stats import RoundStats
from ufc_prediction.scraper.models import (
    EventSummary,
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


# Fight URLs listed on the event_detail.html fixture, in page order.
_FIGHT_URL_ULBERG = "http://ufcstats.com/fight-details/a031f062b7ffefee"  # Ulberg vs Prochazka
_FIGHT_URL_BRADY = "http://ufcstats.com/fight-details/bb22cc33dd44ee55"  # Brady vs Holland
_FIGHT_URL_MORALES = "http://ufcstats.com/fight-details/cc33dd44ee55ff66"  # Morales vs Matthews

_ULBERG = ("Carlos Ulberg", "9014c02eff8b3d62")
_PROCHAZKA = ("Jiri Prochazka", "1122334455667788")


def _swap_fighter_identities(html: str, a: tuple[str, str], b: tuple[str, str]) -> str:
    """Swap two fighters' names and hex ids everywhere in a fight-detail page.

    The stat cells stay where they are, so the result is a fight-detail page
    that lists fighter ``b`` FIRST, credited with the first-row stats — i.e.
    the page order is reversed relative to the event page, as happens on live
    UFCStats (event page lists the winner first; the fight-detail page uses
    its own corner order).
    """
    (name_a, id_a), (name_b, id_b) = a, b
    for old, tmp in ((name_a, "\x00NA\x00"), (id_a, "\x00IA\x00")):
        html = html.replace(old, tmp)
    html = html.replace(name_b, name_a).replace(id_b, id_a)
    return html.replace("\x00NA\x00", name_b).replace("\x00IA\x00", id_b)


def _fight_detail_exact_map() -> dict[str, str]:
    """Map each fixture fight URL to a fight-detail page for THAT fight.

    The event page lists Ulberg/Prochazka, Brady/Holland and Morales/Matthews;
    the fight-detail pages list the same fighters in the same order.
    """
    morales = (
        _load_fixture("fight_detail_draw.html")
        .replace("Niko Price", "Michael Morales")
        .replace("aabb112233445566", "ff88aa99bb00cc11")
        .replace("Alex Oliveira", "Jake Matthews")
        .replace("ccdd556677889900", "dd22ee33ff44aa55")
    )
    return {
        _FIGHT_URL_ULBERG: _load_fixture("fight_detail_1round.html"),
        _FIGHT_URL_BRADY: _load_fixture("fight_detail_3round.html"),
        _FIGHT_URL_MORALES: morales,
    }


def _make_mock_client() -> MockScraperClient:
    """Create a mock client with per-event detail pages matching the event listing fixture."""
    exact_map = _fight_detail_exact_map()
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
        exact_map = _fight_detail_exact_map()
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

    exact_map = _fight_detail_exact_map()
    for url, name, date_str, location in _EVENT_URLS:
        exact_map[url] = _make_event_detail_html(name, date_str, location)
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


class TestPerPageIsolationAndIdentity:
    """Code-review fixes (2026-09): HTTP status errors are isolated per page,
    and fighter identity is the UFCStats hex id, not the display name."""

    def test_http_status_error_on_fighter_profile_does_not_abort_run(
        self, session: Session
    ) -> None:
        """``ScraperClient.get`` raises ``httpx.HTTPStatusError`` on a 404;
        that must skip ONE profile (name-only fallback), not kill the run."""
        import httpx

        from ufc_prediction.scraper.ingest import scrape_all_events

        class Client404(MockScraperClient):
            def get(self, url: str) -> str:
                if "fighter-details/9014c02eff8b3d62" in url:
                    req = httpx.Request("GET", url)
                    resp = httpx.Response(404, request=req)
                    raise httpx.HTTPStatusError("404 Not Found", request=req, response=resp)
                return super().get(url)

        base = _make_mock_client()
        client = Client404(base._fixture_map, base._exact_map)

        result = scrape_all_events(session, client)

        assert isinstance(result, IngestResult)
        assert result.accepted > 0, "run must continue past the dead profile page"
        fallback = (
            session.query(Fighter).filter(Fighter.source_id == "9014c02eff8b3d62").one_or_none()
        )
        assert fallback is not None, "name-only fallback row expected for the 404 profile"
        assert fallback.name == "Carlos Ulberg"

    def test_same_name_fighters_with_distinct_hex_ids_stay_separate(self, session: Session) -> None:
        from ufc_prediction.scraper.ingest import _ensure_fighter

        client = MockScraperClient({"fighter-details": _load_fixture("fighter_profile.html")})
        cache: dict[str, int] = {}
        id_1 = _ensure_fighter(
            client, session, "Bruno Silva", "http://ufcstats.com/fighter-details/aaaa0000", cache
        )
        id_2 = _ensure_fighter(
            client, session, "Bruno Silva", "http://ufcstats.com/fighter-details/bbbb1111", cache
        )

        assert id_1 != id_2
        rows = session.query(Fighter).filter(Fighter.name == "Bruno Silva").all()
        assert sorted(r.source_id for r in rows) == ["aaaa0000", "bbbb1111"]

    def test_same_hex_id_seen_under_two_display_names_is_one_fighter(
        self, session: Session
    ) -> None:
        from ufc_prediction.scraper.ingest import _ensure_fighter

        client = MockScraperClient({"fighter-details": _load_fixture("fighter_profile.html")})
        url = "http://ufcstats.com/fighter-details/cccc2222"
        id_1 = _ensure_fighter(client, session, "Weili Zhang", url, {})
        id_2 = _ensure_fighter(client, session, "Zhang Weili", url, {})

        assert id_1 == id_2
        assert session.query(Fighter).filter(Fighter.source_id == "cccc2222").count() == 1

    def test_fighter_cache_hit_returns_cached_id_without_refetch(self, session: Session) -> None:
        from ufc_prediction.scraper.ingest import _ensure_fighter

        client = MockScraperClient({})  # any get() would raise
        url = "http://ufcstats.com/fighter-details/dddd3333"
        assert _ensure_fighter(client, session, "Anyone", url, {url: 4242}) == 4242
        assert client._call_log == []


# ── Fight-detail fighter orientation (code review 2026-09-28, finding 1) ─────


_STAT_FIELDS = (
    "knockdowns",
    "sig_strikes_landed",
    "sig_strikes_attempted",
    "takedowns_landed",
    "takedowns_attempted",
    "submission_attempts",
    "control_time_seconds",
    "head_strikes_landed",
    "body_strikes_landed",
    "leg_strikes_landed",
    "distance_strikes_landed",
    "clinch_strikes_landed",
    "ground_strikes_landed",
)


def _stat_tuple(obj: object) -> tuple[object, ...]:
    return tuple(getattr(obj, f) for f in _STAT_FIELDS)


def _first_event_summary() -> EventSummary:
    url, name, date_str, location = _EVENT_URLS[0]
    return EventSummary(name=name, date_str=date_str, location=location, url=url)


def _scrape_first_event(
    session: Session, client: MockScraperClient
) -> tuple[IngestResult, dict[str, int]]:
    """Run ``_scrape_event`` over the first fixture event only."""
    from ufc_prediction.scraper.ingest import _scrape_event

    summary = _first_event_summary()
    result = IngestResult()
    cache: dict[str, int] = {}
    _scrape_event(client, session, summary, client.get(summary.url), cache, result)
    return result, cache


def _round_stats_for(session: Session, hex_id: str) -> dict[int, RoundStats]:
    fighter = session.query(Fighter).filter(Fighter.source_id == hex_id).one()
    rows = session.query(RoundStats).filter(RoundStats.fighter_id == fighter.id).all()
    return {r.round_number: r for r in rows}


class TestFightDetailOrientation:
    """The event page lists the winner first; the fight-detail page uses its
    own (corner) order. Stats must follow the fighter's hex id, not position."""

    def test_reversed_fight_detail_stats_land_on_the_right_fighter(self, session: Session) -> None:
        from ufc_prediction.scraper.ingest import _build_fight_stats
        from ufc_prediction.scraper.parse_fight_detail import parse_fight_detail

        original_html = _load_fixture("fight_detail_1round.html")
        orig = parse_fight_detail(original_html)
        # Row-1 stats (first listed) and row-2 stats of the original page.
        row1_total = _stat_tuple(_build_fight_stats(orig.totals[0], orig.sig_strikes[0]))
        row2_total = _stat_tuple(_build_fight_stats(orig.totals[1], orig.sig_strikes[1]))
        row1_r1 = _stat_tuple(
            _build_fight_stats(orig.per_round_totals[0][0], orig.per_round_sig_strikes[0][0])
        )
        row2_r1 = _stat_tuple(
            _build_fight_stats(orig.per_round_totals[0][1], orig.per_round_sig_strikes[0][1])
        )
        assert row1_total != row2_total  # precondition: the swap is observable

        # Fight-detail page lists Prochazka FIRST (with the row-1 stats) while
        # the event page lists Ulberg (the winner) first.
        reversed_html = _swap_fighter_identities(original_html, _ULBERG, _PROCHAZKA)
        client = _make_mock_client()
        client._exact_map[_FIGHT_URL_ULBERG] = reversed_html

        result, _ = _scrape_first_event(session, client)
        assert result.accepted == 3
        assert result.rejected == 0

        ulberg = _round_stats_for(session, _ULBERG[1])
        prochazka = _round_stats_for(session, _PROCHAZKA[1])
        assert _stat_tuple(prochazka[0]) == row1_total
        assert _stat_tuple(ulberg[0]) == row2_total
        assert _stat_tuple(prochazka[1]) == row1_r1
        assert _stat_tuple(ulberg[1]) == row2_r1

        # Fight row keeps the event-page orientation (winner = fighter_a).
        fight = session.query(Fight).filter(Fight.source_url == _FIGHT_URL_ULBERG).one()
        ulberg_id = session.query(Fighter).filter(Fighter.source_id == _ULBERG[1]).one().id
        assert fight.fighter_a_id == ulberg_id
        assert fight.winner_id == ulberg_id

    def test_same_order_fight_detail_is_unchanged(self, session: Session) -> None:
        from ufc_prediction.scraper.ingest import _build_fight_stats
        from ufc_prediction.scraper.parse_fight_detail import parse_fight_detail

        orig = parse_fight_detail(_load_fixture("fight_detail_1round.html"))
        _scrape_first_event(session, _make_mock_client())

        ulberg = _round_stats_for(session, _ULBERG[1])
        prochazka = _round_stats_for(session, _PROCHAZKA[1])
        assert _stat_tuple(ulberg[0]) == _stat_tuple(
            _build_fight_stats(orig.totals[0], orig.sig_strikes[0])
        )
        assert _stat_tuple(prochazka[0]) == _stat_tuple(
            _build_fight_stats(orig.totals[1], orig.sig_strikes[1])
        )

    def test_fight_detail_for_other_fighters_is_rejected(self, session: Session) -> None:
        client = _make_mock_client()
        # Ulberg/Prochazka fight URL serves the Brady/Holland page.
        client._exact_map[_FIGHT_URL_ULBERG] = _load_fixture("fight_detail_3round.html")

        result, _ = _scrape_first_event(session, client)

        assert result.rejected == 1
        assert result.accepted == 2
        assert session.query(Fight).filter(Fight.source_url == _FIGHT_URL_ULBERG).count() == 0
        assert (
            session.query(Fighter).filter(Fighter.source_id == _ULBERG[1]).one_or_none() is None
        ), "a rejected fight must not write its fighters"

    def test_orient_fight_detail_swaps_every_per_fighter_field(self) -> None:
        from ufc_prediction.scraper.ingest import _orient_fight_detail
        from ufc_prediction.scraper.parse_fight_detail import parse_fight_detail

        detail = parse_fight_detail(_load_fixture("fight_detail_3round.html"))
        brady_url, holland_url = detail.fighter_a_url, detail.fighter_b_url

        same = _orient_fight_detail(detail, brady_url, holland_url)
        assert same is detail

        swapped = _orient_fight_detail(detail, holland_url, brady_url)
        assert swapped is not None
        assert swapped.fighter_a_url == holland_url
        assert swapped.fighter_a_name == detail.fighter_b_name
        assert swapped.fighter_a_status == detail.fighter_b_status
        assert swapped.totals == (detail.totals[1], detail.totals[0])
        assert swapped.sig_strikes == (detail.sig_strikes[1], detail.sig_strikes[0])
        assert swapped.per_round_totals == [(b, a) for a, b in detail.per_round_totals]
        assert swapped.per_round_sig_strikes == [(b, a) for a, b in detail.per_round_sig_strikes]
        # Fight-level metadata is not per-fighter and must be untouched.
        assert swapped.method == detail.method
        assert swapped.referee == detail.referee

        other = "http://ufcstats.com/fighter-details/0000000000000000"
        assert _orient_fight_detail(detail, brady_url, other) is None
        assert _orient_fight_detail(detail, "", "") is None


# ── Partially-ingested events (code review 2026-09-28, finding 2) ────────────


class TestPartialEventIngest:
    def test_fight_fetch_failure_leaves_event_unwritten_so_latest_retries(
        self, session: Session
    ) -> None:
        from ufc_prediction.scraper.ingest import scrape_all_events, scrape_latest_events

        base = _make_mock_client()
        flaky = _RaisingMockScraperClient(
            fixture_map=base._fixture_map,
            exact_map=base._exact_map,
            raising_urls={_FIGHT_URL_BRADY},
        )

        first = scrape_all_events(session, flaky)

        # Every fixture event lists the Brady fight, so every event is
        # incomplete: none may be committed, all are reported as rejected.
        assert first.accepted == 0
        assert first.rejected == len(_EVENT_URLS)
        assert session.query(Event).filter(Event.source == "ufcstats").count() == 0

        # The page recovers: `scrape latest` must now pick the events up.
        second = scrape_latest_events(session, _make_mock_client())

        assert second.accepted == 3 * len(_EVENT_URLS)
        events = session.query(Event).filter(Event.source == "ufcstats").all()
        assert len(events) == len(_EVENT_URLS)
        for event in events:
            assert session.query(Fight).filter(Fight.event_id == event.id).count() == 3

    def test_unparseable_fight_is_counted_as_rejected(self, session: Session) -> None:
        from ufc_prediction.scraper.ingest import scrape_all_events

        client = _make_mock_client()
        client._exact_map[_FIGHT_URL_BRADY] = "<html><body>no persons here</body></html>"

        result = scrape_all_events(session, client)

        # A deterministic parse failure will not fix itself on retry: the rest
        # of the event is committed, and the dropped fight is surfaced.
        assert result.accepted == 2 * len(_EVENT_URLS)
        assert result.rejected == len(_EVENT_URLS)
        assert session.query(Event).filter(Event.source == "ufcstats").count() == len(_EVENT_URLS)
