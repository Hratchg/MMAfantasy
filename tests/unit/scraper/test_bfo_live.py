"""Tests for ufc_prediction.scraper.bfo_live (LIVE-01).

Single-shot live BFO matchup-odds fetch for ``ufc predict matchup``. Per
CONTEXT.md D-09: fetch order is cache → live HTTP → NaN.
Per D-10: cache TTL = until event_date passes; ``--refresh`` forces live.
Per Gotcha 1: ``bfo_live`` is a sibling of ``bfo_scraper`` (single-shot,
not batch); reuses parsers, never raises.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
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
    return (
        f"<html><body>"
        f'<a href="/fighters/{slug}-{fid}">{fighter_name}</a>'
        f"</body></html>"
    )


def _real_bfo_fighter_html() -> str:
    """A real captured BFO fighter page from the v1.1 fixtures."""
    return (FIXTURES / "bfo_fighter.html").read_text(encoding="utf-8")


def _captcha_html() -> str:
    """Minimal HTML containing the BFO Cloudflare CAPTCHA marker."""
    return (
        '<html><body><div id="hfmr8">'
        "Please verify you are not a robot</div></body></html>"
    )


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
        fetched_at=datetime.now(timezone.utc),
        source="cache",
    )
    monkeypatch.setattr(
        bfo_live, "_try_cache", lambda session, fa, fb, ed: cached
    )

    client = MagicMock()
    session = MagicMock()
    result = bfo_live.fetch_matchup_odds(
        "A", "B", date(2026, 6, 14), client=client, session=session,
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
        opening=100, closing_range_min=110, closing_range_max=130,
    )
    page = BFOFighterPage(name="A", url="x", fights=[fight])
    monkeypatch.setattr(bfo_live, "parse_bfo_fighter_page", lambda html, url: page)

    client = MagicMock()
    client.get.side_effect = [_make_search_html("A", "A"), "<html>profile</html>"]

    result = bfo_live.fetch_matchup_odds(
        "A", "B", date(2026, 6, 14),
        client=client, session=MagicMock(), refresh=True,
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
        "A", "B", date(2026, 6, 14), client=client, session=None,
    )

    assert result is None


def test_fetch_matchup_odds_runtime_error_returns_none():
    """ScraperClient.get raises RuntimeError (retries exhausted) → None."""
    client = MagicMock()
    client.get.side_effect = RuntimeError("Failed to fetch after 3 retries")

    result = bfo_live.fetch_matchup_odds(
        "A", "B", date(2026, 6, 14), client=client, session=None,
    )

    assert result is None


# ── Test 5: CAPTCHA → None ──────────────────────────────────────────────────


def test_fetch_matchup_odds_captcha_returns_none():
    """Search HTML contains BFO CAPTCHA marker → None (T-15-01-02)."""
    client = MagicMock()
    client.get.return_value = _captcha_html()

    result = bfo_live.fetch_matchup_odds(
        "A", "B", date(2026, 6, 14), client=client, session=None,
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
        "A", "B", date(2026, 6, 14), client=client, session=None,
    )

    assert result is None


# ── Test 7: search miss → None ──────────────────────────────────────────────


def test_fetch_matchup_odds_search_miss_returns_none(monkeypatch):
    """find_bfo_fighter_url returns None (no candidate clears threshold) → None."""
    monkeypatch.setattr(bfo_live, "find_bfo_fighter_url", lambda *a, **kw: None)

    client = MagicMock()
    client.get.return_value = "<html><body>no fighters</body></html>"

    result = bfo_live.fetch_matchup_odds(
        "A", "B", date(2026, 6, 14), client=client, session=None,
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
        opening=100, closing_range_min=110, closing_range_max=130,
    )
    page = BFOFighterPage(name="A", url="x", fights=[fight])
    monkeypatch.setattr(bfo_live, "parse_bfo_fighter_page", lambda html, url: page)

    client = MagicMock()
    client.get.side_effect = [_make_search_html("A", "A"), "<html>profile</html>"]

    result = bfo_live.fetch_matchup_odds(
        "A", "B", date(2026, 6, 14), client=client, session=None,
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

    monkeypatch.setattr(bfo_live, "ScraperClient", FakeClient)

    result = bfo_live.fetch_matchup_odds(
        "A", "B", date(2026, 6, 14), session=None,
    )

    assert result is None
    assert constructed == [1]  # default client was instantiated once
