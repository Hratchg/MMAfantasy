"""Unit tests for the SherdogScraper orchestrator (260423-agz rebuild).

Covers the three new capabilities:

1. ``since_year`` fighter-set filter (via SQL, not in-Python).
2. Incremental ``session.commit()`` every ``commit_batch_size`` matches
   (resilience against mid-run interruption).
3. ``workers`` parameter dispatches batched fetches via ``client.map``.

The parsers (``parse_sherdog_fighter_page``, ``parse_sherdog_search_results``)
and name matcher are NOT exercised here — they have their own test modules
(``test_sherdog_parser.py``, ``test_sherdog_matcher.py``) which must continue
to pass unchanged.
"""

from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import MagicMock

import pytest

from ufc_prediction.scraper import sherdog as sherdog_module
from ufc_prediction.scraper.sherdog import (
    _SHERDOG_SEARCH_URL,
    SherdogScraper,
)
from ufc_prediction.scraper.sherdog_models import SherdogFighterProfile

# ── Test doubles ────────────────────────────────────────────────────────────


class FakeClient:
    """Deterministic test double for ScraperClient.

    - ``get(url)`` returns ``self.responses.get(url, default_html)`` OR
      raises ``RuntimeError`` if ``url in self.raise_urls``.
    - ``map(fn, urls)`` runs INLINE (no threads) and records the url list on
      ``self.map_calls`` (list[list[str]]).
    - ``map_get(urls)`` = ``self.map(self.get, urls)``.
    """

    def __init__(self, default_html: str = "<html></html>") -> None:
        self.get_calls: list[str] = []
        self.map_calls: list[list[str]] = []
        self.raise_urls: set[str] = set()
        self.responses: dict[str, str] = {}
        self._default_html = default_html

    def get(self, url: str) -> str:
        self.get_calls.append(url)
        if url in self.raise_urls:
            raise RuntimeError(f"mock fail for {url}")
        return self.responses.get(url, self._default_html)

    def map(self, fn: Any, urls: Any) -> list[Any]:
        urls_list = list(urls)
        self.map_calls.append(list(urls_list))
        return [fn(u) for u in urls_list]

    def map_get(self, urls: Any) -> list[str]:
        return self.map(self.get, urls)


def _make_fighter(
    fighter_id: int,
    name: str,
    sherdog_url: str | None = None,
) -> MagicMock:
    """Create a lightweight Mock fighter matching the SQLAlchemy Fighter shape."""
    f = MagicMock()
    f.id = fighter_id
    f.name = name
    f.sherdog_url = sherdog_url
    f.pre_ufc_record = None
    return f


def _install_query_stubs(
    monkeypatch: pytest.MonkeyPatch,
    fighters: list[MagicMock],
    first_ufc_dates: dict[int, date | None],
) -> MagicMock:
    """Build a MagicMock session that returns ``fighters`` from the Fighter
    query and ``first_ufc_dates[fighter_id]`` from the min(Event.date) query.

    Also captures every ``filter()`` call chain so tests can assert whether
    the ``since_year`` filter was applied.
    """
    session = MagicMock()
    session.filter_calls: list[Any] = []  # type: ignore[attr-defined]

    # Query chain for the Fighter query.
    # scrape_all_fighters calls either:
    #   self._session.query(Fighter).all()               (no since_year)
    # or
    #   self._session.query(Fighter).filter(...).all()    (with since_year)
    #
    # It also calls self._session.query(func.min(Event.date)) ... .scalar()
    # for each fighter to get first UFC date — we route on call pattern.

    current_fighter_holder = {"idx": 0}

    def query(*args: Any, **kwargs: Any) -> MagicMock:
        # Inspect the first arg to route.
        q = MagicMock()
        first = args[0] if args else None
        # Fighter query: an instance of the Fighter mapped class.
        # In our tests we import Fighter from the real module.
        from ufc_prediction.models.fighter import Fighter

        if first is Fighter:
            # Supports .filter(...).all() and .all() chains.
            def fighter_filter(*_f_args: Any, **_f_kwargs: Any) -> MagicMock:
                session.filter_calls.append(_f_args)  # type: ignore[attr-defined]
                chained = MagicMock()
                chained.all.return_value = fighters
                chained.filter.return_value = chained
                return chained

            q.filter.side_effect = fighter_filter
            q.all.return_value = fighters
            return q

        # min(Event.date) query: returns a query whose .join().filter().scalar()
        # resolves to the first_ufc_date for the next fighter in processing
        # order. We use a shared counter so successive .scalar() calls walk
        # through the fighters list.
        def scalar() -> date | None:
            idx = current_fighter_holder["idx"]
            if idx >= len(fighters):
                return None
            fighter = fighters[idx]
            current_fighter_holder["idx"] += 1
            return first_ufc_dates.get(fighter.id)

        chained = MagicMock()
        chained.join.return_value = chained
        chained.filter.return_value = chained
        chained.scalar.side_effect = scalar
        return chained

    session.query.side_effect = query
    return session


# ── Test 1: since_year filters the query ───────────────────────────────────


def test_since_year_applies_filter_to_fighter_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``since_year=2015`` causes a ``.filter(...)`` call on the Fighter query.

    We don't validate the exact SQL (leave that to integration); we validate
    the BEHAVIOR: when ``since_year`` is provided, the Fighter query receives
    a filter. When it's None, no filter is applied.
    """
    fighter_b = _make_fighter(2, "Fighter B")

    # With since_year: only Fighter B should come back (SQL-filtered).
    session_filtered = _install_query_stubs(
        monkeypatch,
        fighters=[fighter_b],
        first_ufc_dates={2: date(2018, 3, 15)},
    )
    client = FakeClient()

    scraper = SherdogScraper(client, session_filtered)
    result = scraper.scrape_all_fighters(since_year=2015)

    # The Fighter query must have had .filter(...) called on it exactly once.
    assert len(session_filtered.filter_calls) == 1, (  # type: ignore[attr-defined]
        f"expected 1 filter call with since_year, "
        f"got {len(session_filtered.filter_calls)}"  # type: ignore[attr-defined]
    )
    # Only Fighter B's search URL was attempted.
    attempted_searches = [u for u in client.get_calls if "SearchTxt" in u]
    if client.map_calls:
        # workers=1 default uses .get(); but if someone changes default, handle .map() too
        for batch in client.map_calls:
            attempted_searches.extend([u for u in batch if "SearchTxt" in u])
    assert any("Fighter+B" in u for u in attempted_searches), (
        f"expected Fighter B search URL, got {attempted_searches}"
    )
    # skipped should be 0 — A and C were filtered out BEFORE the main loop.
    assert result["skipped"] == 0


def test_since_year_none_skips_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """``since_year=None`` (default) does NOT call ``.filter(...)``."""
    fighter = _make_fighter(1, "Fighter A")
    session = _install_query_stubs(
        monkeypatch,
        fighters=[fighter],
        first_ufc_dates={1: date(2010, 6, 1)},
    )
    client = FakeClient()

    scraper = SherdogScraper(client, session)
    scraper.scrape_all_fighters(since_year=None)

    # No filter was applied on the Fighter query when since_year is None.
    assert session.filter_calls == [], (  # type: ignore[attr-defined]
        f"expected 0 filter calls without since_year, "
        f"got {session.filter_calls}"  # type: ignore[attr-defined]
    )


# ── Test 2: incremental commit fires every N fighters ─────────────────────


def test_incremental_commit_fires_every_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``commit_batch_size=3`` and 7 successful matches, at least one
    intermediate commit must have fired (i.e. ``.commit()`` called >= 2 times,
    proving partial-progress resilience).
    """
    fighters = [_make_fighter(i, f"Fighter {i}") for i in range(1, 8)]
    first_ufc_dates = {i: date(2020, 1, 1) for i in range(1, 8)}

    session = _install_query_stubs(monkeypatch, fighters, first_ufc_dates)
    client = FakeClient()

    # Stub parse_sherdog_search_results to return one candidate per search
    # call, matching the fighter's name exactly so match_fighter_name succeeds.
    search_call_counter = {"i": 0}

    def fake_parse_search(_html: str) -> list[tuple[str, str]]:
        idx = search_call_counter["i"]
        search_call_counter["i"] += 1
        fighter = fighters[idx] if idx < len(fighters) else fighters[-1]
        return [(fighter.name, f"https://www.sherdog.com/fighter/fake-{fighter.id}")]

    monkeypatch.setattr(
        sherdog_module, "parse_sherdog_search_results", fake_parse_search
    )

    # Use dry_run=True so we skip profile-fetch path entirely but still
    # go through the matched + _maybe_commit code.
    scraper = SherdogScraper(client, session, commit_batch_size=3)
    result = scraper.scrape_all_fighters(dry_run=True)

    # 7 matches with commit_batch_size=3 → expect >= 2 commits total
    # (at least one intermediate fire + final flush).
    assert session.commit.call_count >= 2, (
        f"expected >= 2 commits with 7 fighters/batch=3, "
        f"got {session.commit.call_count}"
    )
    assert result["matched"] == 7


# ── Test 3: workers=2 dispatches via client.map ───────────────────────────


def test_workers_gt_1_uses_client_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``workers=2`` and 4 fighters, ``client.map`` is called for
    batched fetches (at minimum the search stage). Each recorded URL list
    has len <= 2 (the chunk size)."""
    fighters = [_make_fighter(i, f"Fighter {i}") for i in range(1, 5)]
    first_ufc_dates = {i: date(2020, 1, 1) for i in range(1, 5)}

    session = _install_query_stubs(monkeypatch, fighters, first_ufc_dates)
    client = FakeClient()

    # Stub parsers so matching succeeds; give profile parser a minimal
    # SherdogFighterProfile.
    def fake_parse_search(_html: str) -> list[tuple[str, str]]:
        # Return a name that matches any of our Fighter N fighters (use
        # generic "Fighter" prefix; matcher is fuzzy enough)
        return [("Fighter 1", "https://www.sherdog.com/fighter/fake")]

    def fake_parse_profile(_html: str, url: str) -> SherdogFighterProfile:
        return SherdogFighterProfile(name="Fighter 1", sherdog_url=url, fights=[])

    def fake_match(
        db_name: str, candidates: list[tuple[str, str]], threshold: int = 85
    ) -> tuple[str, str] | None:
        # Always match the first candidate — we're testing dispatch, not fuzz logic.
        return candidates[0] if candidates else None

    monkeypatch.setattr(
        sherdog_module, "parse_sherdog_search_results", fake_parse_search
    )
    monkeypatch.setattr(
        sherdog_module, "parse_sherdog_fighter_page", fake_parse_profile
    )
    monkeypatch.setattr(sherdog_module, "match_fighter_name", fake_match)

    scraper = SherdogScraper(client, session, workers=2)
    scraper.scrape_all_fighters()

    # At least one .map() call was made (probably multiple: search stage
    # + profile stage per chunk).
    assert len(client.map_calls) >= 2, (
        f"expected at least 2 .map() calls (search + profile), "
        f"got {len(client.map_calls)}"
    )
    # Every recorded URL batch has len <= workers (2).
    for batch in client.map_calls:
        assert len(batch) <= 2, (
            f"batch exceeded workers=2 chunk size: {batch}"
        )


# ── Test 4: workers=1 preserves legacy serial behavior ────────────────────


def test_workers_1_uses_serial_get_no_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``workers=1`` (default), ``client.map`` is never called and the
    scraper uses serial ``client.get()`` in the legacy pattern (2 calls per
    matched fighter: search + profile)."""
    fighters = [_make_fighter(i, f"Fighter {i}") for i in range(1, 4)]
    first_ufc_dates = {i: date(2020, 1, 1) for i in range(1, 4)}

    session = _install_query_stubs(monkeypatch, fighters, first_ufc_dates)
    client = FakeClient()

    def fake_parse_search(_html: str) -> list[tuple[str, str]]:
        return [("Fighter 1", "https://www.sherdog.com/fighter/fake")]

    def fake_parse_profile(_html: str, url: str) -> SherdogFighterProfile:
        return SherdogFighterProfile(name="Fighter 1", sherdog_url=url, fights=[])

    def fake_match(
        db_name: str, candidates: list[tuple[str, str]], threshold: int = 85
    ) -> tuple[str, str] | None:
        return candidates[0] if candidates else None

    monkeypatch.setattr(
        sherdog_module, "parse_sherdog_search_results", fake_parse_search
    )
    monkeypatch.setattr(
        sherdog_module, "parse_sherdog_fighter_page", fake_parse_profile
    )
    monkeypatch.setattr(sherdog_module, "match_fighter_name", fake_match)

    scraper = SherdogScraper(client, session, workers=1)
    result = scraper.scrape_all_fighters()

    # No map() calls on the workers=1 path.
    assert client.map_calls == [], (
        f"workers=1 path should not call .map(), got {client.map_calls}"
    )
    # 2 .get() calls per matched fighter (search + profile).
    assert len(client.get_calls) == 6, (
        f"expected 2 get calls per fighter x 3 fighters = 6, "
        f"got {len(client.get_calls)}"
    )
    # Outcome counts look sane.
    assert result["matched"] == 3
    assert result["unmatched"] == 0
    assert result["errors"] == 0
    assert result["skipped"] == 0


# ── Test 5: per-URL fetch failure is isolated (workers=2) ─────────────────


def test_workers_gt_1_isolates_per_url_fetch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With workers>1, a RuntimeError on a single URL in a batch causes
    exactly that fighter to be counted in ``errors``; other fighters in
    the same chunk still match successfully and no exception escapes."""
    fighters = [_make_fighter(i, f"Fighter {i}") for i in range(1, 3)]
    first_ufc_dates = {i: date(2020, 1, 1) for i in range(1, 3)}

    session = _install_query_stubs(monkeypatch, fighters, first_ufc_dates)
    client = FakeClient()

    # Make Fighter 1's SEARCH URL fail; Fighter 2's succeed.
    fighter1_search = _SHERDOG_SEARCH_URL.format(name="Fighter+1")
    client.raise_urls.add(fighter1_search)

    def fake_parse_search(_html: str) -> list[tuple[str, str]]:
        return [("Fighter 2", "https://www.sherdog.com/fighter/fake-2")]

    def fake_parse_profile(_html: str, url: str) -> SherdogFighterProfile:
        return SherdogFighterProfile(name="Fighter 2", sherdog_url=url, fights=[])

    def fake_match(
        db_name: str, candidates: list[tuple[str, str]], threshold: int = 85
    ) -> tuple[str, str] | None:
        return candidates[0] if candidates else None

    monkeypatch.setattr(
        sherdog_module, "parse_sherdog_search_results", fake_parse_search
    )
    monkeypatch.setattr(
        sherdog_module, "parse_sherdog_fighter_page", fake_parse_profile
    )
    monkeypatch.setattr(sherdog_module, "match_fighter_name", fake_match)

    scraper = SherdogScraper(client, session, workers=2)
    result = scraper.scrape_all_fighters()

    # Fighter 1 → errors (search fetch failed for its URL).
    # Fighter 2 → matched (succeeded through the whole pipeline).
    assert result["errors"] == 1, f"expected errors=1, got {result}"
    assert result["matched"] == 1, f"expected matched=1, got {result}"


# ── Test 6: workers validation ────────────────────────────────────────────


def test_workers_lt_1_raises() -> None:
    """``workers < 1`` is rejected with ValueError at construction time."""
    with pytest.raises(ValueError, match="workers"):
        SherdogScraper(MagicMock(), MagicMock(), workers=0)


def test_commit_batch_size_lt_1_raises() -> None:
    """``commit_batch_size < 1`` is rejected."""
    with pytest.raises(ValueError, match="commit_batch_size"):
        SherdogScraper(MagicMock(), MagicMock(), commit_batch_size=0)
