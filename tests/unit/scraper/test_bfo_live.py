"""Tests for ufc_prediction.scraper.bfo_live (LIVE-01).

Single-shot live BFO matchup-odds fetch for ``ufc predict matchup``. Per
CONTEXT.md D-09: fetch order is cache → live HTTP → NaN.
Per D-10: cache TTL = until event_date passes; ``--refresh`` forces live.
Per Gotcha 1: ``bfo_live`` is a sibling of ``bfo_scraper`` (single-shot,
not batch); reuses parsers, never raises.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

bfo_live = pytest.importorskip("ufc_prediction.scraper.bfo_live")

FIXTURES = Path(__file__).resolve().parents[2] / "scraper" / "fixtures"


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_search_html(fighter_name: str, slug: str, fid: str = "999") -> str:
    """Build a minimal BFO search HTML page that ``find_bfo_fighter_url``
    can resolve to ``/fighters/<slug>-<fid>``.

    Real BFO search returns a table of fighter links; we mimic the
    smallest shape ``find_bfo_fighter_url`` will accept (it scans
    ``a[href^='/fighters/']`` and fuzzy-matches the link text).
    """
    return f'<html><body><a href="/fighters/{slug}-{fid}">{fighter_name}</a></body></html>'


def _real_bfo_fighter_html() -> str:
    """A real captured BFO fighter page from the v1.1 fixtures."""
    return (FIXTURES / "bfo_fighter.html").read_text(encoding="utf-8")


def _captcha_html() -> str:
    """Minimal HTML containing the BFO Cloudflare CAPTCHA marker."""
    return '<html><body><div id="hfmr8">Please verify you are not a robot</div></body></html>'


# ── Test 1: happy-path live fetch ───────────────────────────────────────────


def test_fetch_matchup_odds_live_happy_path(monkeypatch):
    """Cache miss → live HTTP succeeds → returns MatchupOdds(source='live').

    Mocks ``ScraperClient.get`` (search + profile) and patches
    ``parse_bfo_fighter_page`` to return a synthetic ``BFOFighterPage``
    whose ``.fights`` contains the matchup we ask for.
    """
    from ufc_prediction.scraper.bfo_scraper import BFOFighterPage, BFOParsedFight

    fight = BFOParsedFight(
        event_date=date(2026, 6, 14),
        opponent_name="Conor McGregor",
        opponent_bfo_id="123",
        opening=-200,
        closing_range_min=-220,
        closing_range_max=-180,
        opponent_opening=170,
        opponent_closing_range_min=160,
        opponent_closing_range_max=190,
    )
    page = BFOFighterPage(name="Khabib Nurmagomedov", url="x", fights=[fight])

    client = MagicMock()
    client.get.side_effect = [
        _make_search_html("Khabib Nurmagomedov", "Khabib-Nurmagomedov"),
        "<html>profile-html</html>",
    ]

    # Patch the parser used inside bfo_live so we don't need real BFO HTML
    monkeypatch.setattr(bfo_live, "parse_bfo_fighter_page", lambda html, url: page)

    result = bfo_live.fetch_matchup_odds(
        "Khabib Nurmagomedov",
        "Conor McGregor",
        date(2026, 6, 14),
        client=client,
        session=None,
    )

    assert result is not None
    assert result.source == "live"
    assert result.fighter_a_opening == -200
    assert result.fighter_a_closing_min == -220
    assert result.fighter_a_closing_max == -180
    assert result.fighter_b_opening == 170
    assert result.fighter_b_closing_min == 160
    assert result.fighter_b_closing_max == 190
    # fetched_at is a real timezone-aware datetime
    assert isinstance(result.fetched_at, datetime)
    assert result.fetched_at.tzinfo is not None
    # Two HTTP calls (search + profile)
    assert client.get.call_count == 2


# ── Test 2: cache hit (no HTTP) ─────────────────────────────────────────────


def test_fetch_matchup_odds_cache_hit_skips_http(monkeypatch):
    """Cache returns a hit → MatchupOdds(source='cache'); client.get NEVER called.

    Patches ``_try_cache`` directly so we don't need a SQLAlchemy session
    plumbed; the contract under test is the dispatch order in
    ``fetch_matchup_odds``, not the cache implementation details.
    """
    from ufc_prediction.scraper.bfo_live import MatchupOdds

    cached = MatchupOdds(
        fighter_a_opening=-150,
        fighter_a_closing_min=-160,
        fighter_a_closing_max=-140,
        fighter_b_opening=130,
        fighter_b_closing_min=120,
        fighter_b_closing_max=140,
        fetched_at=datetime.now(UTC),
        source="cache",
    )
    monkeypatch.setattr(bfo_live, "_try_cache", lambda session, fa, fb, ed: cached)

    client = MagicMock()
    session = MagicMock()
    result = bfo_live.fetch_matchup_odds(
        "A",
        "B",
        date(2026, 6, 14),
        client=client,
        session=session,
    )

    assert result is not None
    assert result.source == "cache"
    assert result.fighter_a_opening == -150
    client.get.assert_not_called()


# ── Test 3: refresh=True bypasses cache ─────────────────────────────────────


def test_fetch_matchup_odds_refresh_bypasses_cache(monkeypatch):
    """refresh=True → cache lookup is SKIPPED; live HTTP IS used."""
    from ufc_prediction.scraper.bfo_scraper import BFOFighterPage, BFOParsedFight

    cached_calls: list = []

    def fake_cache(*args, **kwargs):  # pragma: no cover — must NOT be called
        cached_calls.append(1)
        return MagicMock(source="cache")

    monkeypatch.setattr(bfo_live, "_try_cache", fake_cache)

    fight = BFOParsedFight(
        event_date=date(2026, 6, 14),
        opponent_name="B",
        opponent_bfo_id="2",
        opening=100,
        closing_range_min=110,
        closing_range_max=130,
        opponent_opening=-120,
        opponent_closing_range_min=-150,
        opponent_closing_range_max=-130,
    )
    page = BFOFighterPage(name="A", url="x", fights=[fight])
    monkeypatch.setattr(bfo_live, "parse_bfo_fighter_page", lambda html, url: page)

    client = MagicMock()
    client.get.side_effect = [_make_search_html("A", "A"), "<html>profile</html>"]

    result = bfo_live.fetch_matchup_odds(
        "A",
        "B",
        date(2026, 6, 14),
        client=client,
        session=MagicMock(),
        refresh=True,
    )

    assert result is not None
    assert result.source == "live"
    assert cached_calls == []  # cache lookup skipped per refresh=True
    assert client.get.call_count == 2


# ── Test 4: timeout / transport error → None ────────────────────────────────


def test_fetch_matchup_odds_timeout_returns_none():
    """ScraperClient.get raises a transport error → returns None, never raises."""
    client = MagicMock()
    client.get.side_effect = httpx.TimeoutException("read timeout")

    result = bfo_live.fetch_matchup_odds(
        "A",
        "B",
        date(2026, 6, 14),
        client=client,
        session=None,
    )

    assert result is None


def test_fetch_matchup_odds_runtime_error_returns_none():
    """ScraperClient.get raises RuntimeError (retries exhausted) → None."""
    client = MagicMock()
    client.get.side_effect = RuntimeError("Failed to fetch after 3 retries")

    result = bfo_live.fetch_matchup_odds(
        "A",
        "B",
        date(2026, 6, 14),
        client=client,
        session=None,
    )

    assert result is None


# ── Test 5: CAPTCHA → None ──────────────────────────────────────────────────


def test_fetch_matchup_odds_captcha_returns_none():
    """Search HTML contains BFO CAPTCHA marker → None (T-15-01-02)."""
    client = MagicMock()
    client.get.return_value = _captcha_html()

    result = bfo_live.fetch_matchup_odds(
        "A",
        "B",
        date(2026, 6, 14),
        client=client,
        session=None,
    )

    assert result is None


# ── Test 6: parse error → None ──────────────────────────────────────────────


def test_fetch_matchup_odds_parse_error_returns_none(monkeypatch):
    """parse_bfo_fighter_page raises BFOParseError → None."""
    from ufc_prediction.scraper.bfo_scraper import BFOParseError

    def boom(html, url):
        raise BFOParseError("malformed HTML")

    monkeypatch.setattr(bfo_live, "parse_bfo_fighter_page", boom)

    client = MagicMock()
    client.get.side_effect = [
        _make_search_html("A", "A"),
        "<html>broken-profile</html>",
    ]

    result = bfo_live.fetch_matchup_odds(
        "A",
        "B",
        date(2026, 6, 14),
        client=client,
        session=None,
    )

    assert result is None


# ── Test 7: search miss → None ──────────────────────────────────────────────


def test_fetch_matchup_odds_search_miss_returns_none(monkeypatch):
    """find_bfo_fighter_url returns None (no candidate clears threshold) → None."""
    monkeypatch.setattr(bfo_live, "find_bfo_fighter_url", lambda *a, **kw: None)

    client = MagicMock()
    client.get.return_value = "<html><body>no fighters</body></html>"

    result = bfo_live.fetch_matchup_odds(
        "A",
        "B",
        date(2026, 6, 14),
        client=client,
        session=None,
    )

    assert result is None
    # Only the search request happens; no profile fetch is attempted
    assert client.get.call_count == 1


# ── Test 8: opponent / event-date miss inside profile → None ────────────────


def test_fetch_matchup_odds_no_matching_row_returns_none(monkeypatch):
    """Profile has fights but none match the (opponent_name, event_date) → None."""
    from ufc_prediction.scraper.bfo_scraper import BFOFighterPage, BFOParsedFight

    fight = BFOParsedFight(
        event_date=date(2025, 1, 1),  # different date
        opponent_name="Some Other Guy",
        opponent_bfo_id="1",
        opening=100,
        closing_range_min=110,
        closing_range_max=130,
    )
    page = BFOFighterPage(name="A", url="x", fights=[fight])
    monkeypatch.setattr(bfo_live, "parse_bfo_fighter_page", lambda html, url: page)

    client = MagicMock()
    client.get.side_effect = [_make_search_html("A", "A"), "<html>profile</html>"]

    result = bfo_live.fetch_matchup_odds(
        "A",
        "B",
        date(2026, 6, 14),
        client=client,
        session=None,
    )

    assert result is None


# ── Test 9: client defaults to a fresh ScraperClient ────────────────────────


def test_fetch_matchup_odds_default_client_constructed(monkeypatch):
    """client=None → ScraperClient is constructed lazily inside fetch_matchup_odds."""
    constructed: list = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            constructed.append(1)

        def get(self, url):
            raise httpx.TimeoutException("timeout")

        def close(self):
            pass

    monkeypatch.setattr(bfo_live, "ScraperClient", FakeClient)

    result = bfo_live.fetch_matchup_odds(
        "A",
        "B",
        date(2026, 6, 14),
        session=None,
    )

    assert result is None
    assert constructed == [1]  # default client was instantiated once


# ── Unmocked parse → _try_live → _populate_odds (review finding S10-1) ──────


def _jones_live(event_date: date, fb_name: str = "Alex Pereira"):
    """Run the real parser on the captured Jon Jones fixture via ``_try_live``.

    Only the HTTP layer is mocked (search page + profile page); the BFO
    parser, the row matcher and the B-side extraction all run for real.
    """
    client = MagicMock()
    client.get.side_effect = [
        _make_search_html("Jon Jones", "Jon-Jones", "819"),
        _real_bfo_fighter_html(),
    ]
    return bfo_live._try_live(client, "Jon Jones", fb_name, event_date)


def test_try_live_upcoming_bout_matches_future_events_row():
    """An upcoming bout (event_date >= today) must match BFO's 'Future Events'
    row, which the parser dates as the ``date.max`` sentinel, and must carry
    BOTH sides' moneylines (Jones -250 / Pereira +210)."""
    result = _jones_live(date.today() + timedelta(days=7))

    assert result is not None
    assert result.source == "live"
    assert result.fighter_a_opening == -250
    assert result.fighter_a_closing_min == -250
    assert result.fighter_a_closing_max == -250
    assert result.fighter_b_opening == 210
    assert result.fighter_b_closing_min == 210
    assert result.fighter_b_closing_max == 210


def test_try_live_past_date_exact_row_has_both_sides():
    """A dated (past) row matches on exact date and fills the B side."""
    result = _jones_live(date(2025, 12, 31))

    assert result is not None
    assert (result.fighter_a_opening, result.fighter_a_closing_min) == (-286, -525)
    assert result.fighter_a_closing_max == -286
    assert (result.fighter_b_opening, result.fighter_b_closing_min) == (210, 210)
    assert result.fighter_b_closing_max == 410


def test_try_live_past_date_does_not_match_future_sentinel_row():
    """A past event_date must NOT fall onto an undated 'Future Events' row."""
    assert _jones_live(date(2020, 1, 1)) is None


def test_try_live_upcoming_bout_populates_all_odds_features():
    """End-to-end: the live result drives every one of the 5 odds features."""
    from ufc_prediction.ml.inference_features import _populate_odds

    live = _jones_live(date.today() + timedelta(days=7))
    assert live is not None

    feats: dict[str, float] = dict.fromkeys(
        (
            "opening_prob_diff",
            "closing_prob_diff",
            "line_movement_diff",
            "sharp_money_signal",
            "odds_elo_divergence",
        ),
        float("nan"),
    )
    _populate_odds(feats, live, None, None, 1600.0, 1500.0)

    for key, value in feats.items():
        assert not math.isnan(value), f"{key} stayed NaN on a live hit"
    assert feats["opening_prob_diff"] > 0  # Jones is the favourite


def test_try_live_one_sided_row_returns_none_so_cache_can_fill(monkeypatch):
    """A row with no B-side lines cannot produce any odds feature, so the live
    path must report a miss (``None``) rather than a useless partial hit that
    would suppress the cached-odds fallback in ``inference_features.build``."""
    from ufc_prediction.scraper.bfo_scraper import BFOFighterPage, BFOParsedFight

    fight = BFOParsedFight(
        event_date=date(2026, 6, 14),
        opponent_name="B",
        opponent_bfo_id="2",
        opening=-150,
        closing_range_min=-160,
        closing_range_max=-140,
    )
    page = BFOFighterPage(name="A", url="x", fights=[fight])
    monkeypatch.setattr(bfo_live, "parse_bfo_fighter_page", lambda html, url: page)

    client = MagicMock()
    client.get.side_effect = [_make_search_html("A", "A"), "<html>profile</html>"]

    assert bfo_live._try_live(client, "A", "B", date(2026, 6, 14)) is None


# ── Default predict-time client is bounded + closed (review finding S10-2) ──


def test_default_client_is_fail_fast_and_closed(monkeypatch):
    """client=None → the constructed client has no retries / no inter-request
    delay (so a failing BFO costs ~2x5s, not ~1 min) and is closed after use."""
    instances: list = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            instances.append(self)

        def get(self, url):
            raise RuntimeError("503")

        def close(self):
            self.closed = True

    monkeypatch.setattr(bfo_live, "ScraperClient", FakeClient)

    assert bfo_live.fetch_matchup_odds("A", "B", date(2026, 6, 14), session=None) is None
    assert len(instances) == 1
    kwargs = instances[0].kwargs
    assert kwargs.get("timeout") == 5.0
    assert kwargs.get("max_retries") == 0
    assert kwargs.get("delay") == 0
    assert instances[0].closed is True


def test_caller_supplied_client_is_not_closed():
    """A client passed in by the caller is the caller's to close."""
    client = MagicMock()
    client.get.side_effect = RuntimeError("boom")

    bfo_live.fetch_matchup_odds("A", "B", date(2026, 6, 14), client=client, session=None)

    client.close.assert_not_called()
