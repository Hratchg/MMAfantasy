"""Unit tests for the Playwright-backed :class:`BrowserFetcher`.

Playwright is fully mocked — no real browser is launched and no network is
touched. The tests verify the fetcher's contract:
- returns rendered page content on a clean (non-challenge) page,
- retries then HALTs honestly (raises) on a persistent anti-bot challenge,
- reuses the browser/context/page across calls (solve once),
- honors the proxy constructor arg / env override,
- exposes a serial, order-preserving ``map``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ufc_prediction.scraper.browser_fetch import (
    AntiBotChallengeError,
    BrowserFetcher,
    _parse_proxy,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load(name: str) -> str:
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


class FakePage:
    """Minimal stand-in for a Playwright ``Page``.

    ``contents`` is a list of HTML strings returned by successive ``content()``
    calls (the last entry repeats). ``selector_found`` controls whether
    ``wait_for_selector`` succeeds.
    """

    def __init__(
        self,
        contents: list[str],
        status: int = 200,
        selector_found: bool = True,
    ) -> None:
        self._contents = list(contents)
        self._status = status
        self._selector_found = selector_found
        self.goto_calls: list[str] = []
        self.closed = False

    def goto(self, url: str, **_kwargs: object) -> MagicMock:
        self.goto_calls.append(url)
        resp = MagicMock()
        resp.status = self._status
        return resp

    def wait_for_load_state(self, *_a: object, **_k: object) -> None:
        return None

    def wait_for_selector(self, *_a: object, **_k: object) -> MagicMock:
        if not self._selector_found:
            msg = "selector timeout"
            raise TimeoutError(msg)
        return MagicMock()

    def content(self) -> str:
        if len(self._contents) > 1:
            return self._contents.pop(0)
        return self._contents[0]

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralize all real waiting so tests run instantly."""
    monkeypatch.setattr(
        "ufc_prediction.scraper.browser_fetch.time.sleep",
        lambda _s: None,
    )


def _fetcher_with_page(page: FakePage, **kwargs: object) -> BrowserFetcher:
    fetcher = BrowserFetcher(delay=0.0, **kwargs)  # type: ignore[arg-type]
    fetcher._page = page  # pre-inject so _ensure_browser skips launch
    return fetcher


# ── get() success ────────────────────────────────────────────────────────


def test_get_returns_rendered_content() -> None:
    html = _load("event_list_snippet.html")
    page = FakePage([html])
    fetcher = _fetcher_with_page(page)

    result = fetcher.get("http://ufcstats.com/statistics/events/completed?page=all")

    assert result == html
    assert "b-statistics__table-events" in result
    assert page.goto_calls == [
        "http://ufcstats.com/statistics/events/completed?page=all",
    ]


def test_get_succeeds_on_first_attempt_no_retry() -> None:
    page = FakePage([_load("event_list_snippet.html")])
    fetcher = _fetcher_with_page(page, max_retries=3)

    fetcher.get("http://ufcstats.com/x")

    assert len(page.goto_calls) == 1  # no retries when the page is clean


# ── get() HALTs honestly on persistent challenge ──────────────────────────


def test_get_halts_after_retries_on_challenge() -> None:
    challenge = _load("ufcstats_antibot_challenge.html")
    page = FakePage([challenge])
    fetcher = _fetcher_with_page(page, max_retries=2)

    with pytest.raises(AntiBotChallengeError) as exc:
        fetcher.get("http://ufcstats.com/x")

    # 1 initial attempt + 2 retries = 3 navigations, then HALT.
    assert len(page.goto_calls) == 3
    assert "HALTing honestly" in str(exc.value)


def test_get_retries_then_succeeds() -> None:
    """A challenge on the first hit, then the real page: returns real content."""
    challenge = _load("ufcstats_antibot_challenge.html")
    real = _load("event_list_snippet.html")
    page = FakePage([challenge, real])
    fetcher = _fetcher_with_page(page, max_retries=3)

    result = fetcher.get("http://ufcstats.com/x")

    assert result == real
    assert len(page.goto_calls) == 2


def test_missing_selector_does_not_halt_when_clean() -> None:
    """A missing settling selector on clean HTML is non-fatal (no HALT).

    HALT is driven only by ``detect_antibot`` — the selector is a best-effort
    settling hint. A detail page with no challenge markers must return content
    even if the (listing-only) selector never appears.
    """
    page = FakePage(
        ["<html><body>real fight page</body></html>"],
        selector_found=False,
    )
    fetcher = _fetcher_with_page(page, max_retries=2)

    result = fetcher.get("http://ufcstats.com/fight-details/abc")

    assert "real fight page" in result
    assert len(page.goto_calls) == 1  # returned immediately, no retries


def test_get_halts_when_required_selector_never_renders() -> None:
    """Listing page: clean-looking HTML but the events table never appears.

    A required (URL-derived) selector that never renders means the challenge
    likely did not clear, so the fetcher retries then HALTs — even without a
    Cloudflare signature in the HTML.
    """
    page = FakePage(
        ["<html><body>partial</body></html>"],
        selector_found=False,
    )
    fetcher = _fetcher_with_page(page, max_retries=1)

    with pytest.raises(AntiBotChallengeError):
        fetcher.get("http://ufcstats.com/statistics/events/completed?page=all")

    assert len(page.goto_calls) == 2  # 1 attempt + 1 retry, then HALT


def test_get_skips_selector_wait_when_none() -> None:
    """wait_selector=None returns clean HTML without waiting on any selector."""
    page = FakePage(["<html><body>ok</body></html>"], selector_found=False)
    fetcher = _fetcher_with_page(page, max_retries=1)

    result = fetcher.get("http://ufcstats.com/fight-details/abc", wait_selector=None)

    assert "ok" in result
    assert len(page.goto_calls) == 1


# ── Session / context reuse (solve once) ──────────────────────────────────


def test_browser_launched_once_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    page = FakePage([_load("event_list_snippet.html")])
    fetcher = BrowserFetcher(delay=0.0)

    launch = MagicMock(side_effect=lambda: setattr(fetcher, "_page", page))
    monkeypatch.setattr(fetcher, "_launch", launch)

    fetcher.get("http://ufcstats.com/a")
    fetcher.get("http://ufcstats.com/b")

    assert launch.call_count == 1  # context/cookies solved once, reused
    assert page.goto_calls == ["http://ufcstats.com/a", "http://ufcstats.com/b"]


# ── map: serial, order-preserving ─────────────────────────────────────────


def test_map_is_serial_and_order_preserving() -> None:
    page = FakePage([_load("event_list_snippet.html")])
    fetcher = _fetcher_with_page(page)

    urls = ["http://ufcstats.com/1", "http://ufcstats.com/2", "http://ufcstats.com/3"]
    results = fetcher.map(fetcher.get, urls)

    assert len(results) == 3
    assert page.goto_calls == urls  # serial, in input order


# ── proxy resolution ──────────────────────────────────────────────────────


def test_proxy_constructor_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UFC_SCRAPE_PROXY", "http://env-proxy:1111")
    fetcher = BrowserFetcher(proxy="http://arg-proxy:2222")
    assert fetcher._proxy_url == "http://arg-proxy:2222"


def test_proxy_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UFC_SCRAPE_PROXY", "http://env-proxy:1111")
    fetcher = BrowserFetcher()
    assert fetcher._proxy_url == "http://env-proxy:1111"


def test_parse_proxy_with_credentials() -> None:
    assert _parse_proxy("http://user:pass@1.2.3.4:8080") == {
        "server": "http://1.2.3.4:8080",
        "username": "user",
        "password": "pass",
    }


def test_parse_proxy_none() -> None:
    assert _parse_proxy(None) is None


# ── teardown ──────────────────────────────────────────────────────────────


def test_close_is_idempotent() -> None:
    page = FakePage([_load("event_list_snippet.html")])
    fetcher = _fetcher_with_page(page)
    fetcher.get("http://ufcstats.com/x")

    fetcher.close()
    assert page.closed is True
    fetcher.close()  # second close must not raise
