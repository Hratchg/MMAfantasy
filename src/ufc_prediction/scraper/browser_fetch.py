"""Headless-browser fetcher for UFCStats (JS proof-of-work bypass).

UFCStats.com is gated behind a JavaScript proof-of-work / Cloudflare anti-bot
challenge that a plain ``httpx`` client (``ScraperClient``) cannot solve — it
only receives the challenge stub, never the real page. This module provides a
drop-in replacement that runs a real headless Chromium (via Playwright) so the
challenge JS actually executes and the real content is returned.

The interface mirrors the subset of :class:`ScraperClient` that
``scraper/ingest.py`` uses — ``get(url) -> str``, ``map(fn, urls) -> list``,
``close()``, and the context-manager protocol — so a ``BrowserFetcher`` can be
injected into ``scrape_all_events`` / ``scrape_latest_events`` unchanged.

Politeness / ethics posture (operator-authorized, see KNOWN_ISSUES.md):
- Single worker, serial ``map`` (no parallel hammering).
- >= ``delay`` seconds between requests (default 1.5s), matching
  ``ScraperClient``'s rate-limit pattern.
- The browser context (and any ``cf_clearance`` cookie) is solved ONCE and
  reused across calls — we do not re-solve per request.
- On a persistent challenge after retries we HALT honestly by raising
  :class:`AntiBotChallengeError` — we NEVER fabricate or return stub data.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, cast
from urllib.parse import urlparse

from ufc_prediction.scraper.antibot import detect_antibot

if TYPE_CHECKING:  # pragma: no cover - typing only
    from types import TracebackType

    from playwright.sync_api import ProxySettings, ViewportSize

logger = logging.getLogger(__name__)

T = TypeVar("T")

# A realistic desktop Chrome UA/viewport so the fingerprint looks human. The
# research User-Agent used by ``ScraperClient`` trivially fails the challenge.
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)
_VIEWPORT = {"width": 1440, "height": 900}

# Content selector proving the real UFCStats events listing rendered (as
# opposed to a Cloudflare challenge stub). Used as a best-effort settling wait
# so the proof-of-work JS has time to redirect/render before we read content.
_EVENTS_TABLE_SELECTOR = "table.b-statistics__table-events"

# Sentinel so callers can pass ``wait_selector=None`` explicitly (skip the
# wait) and be distinguished from "caller did not specify — pick a default".
_UNSET = object()


def _default_selector_for(url: str) -> str | None:
    """Pick a best-effort settling selector for a UFCStats URL.

    Only the events *listing* page has a reliable, distinctive table selector.
    For detail/fighter pages we rely on ``networkidle`` + anti-bot detection
    (the shared Cloudflare clearance from the already-solved context means
    these pages return real content without re-challenging).
    """
    if "statistics/events" in url:
        return _EVENTS_TABLE_SELECTOR
    return None


# JS injected before any page script runs to mask the most common headless
# tells (``navigator.webdriver`` etc.).
_STEALTH_INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
window.chrome = window.chrome || { runtime: {} };
"""


class AntiBotChallengeError(RuntimeError):
    """Raised when the anti-bot challenge persists after all retries.

    Surfacing this (rather than returning stub HTML) enforces the HALT-honestly
    contract: a caller must never mistake a challenge page for real data.
    """


def _parse_proxy(proxy_url: str | None) -> dict[str, str] | None:
    """Convert a ``scheme://user:pass@host:port`` URL to Playwright's proxy dict.

    Returns ``None`` when ``proxy_url`` is falsy.
    """
    if not proxy_url:
        return None
    parsed = urlparse(proxy_url)
    server = f"{parsed.scheme}://{parsed.hostname}"
    if parsed.port:
        server = f"{server}:{parsed.port}"
    proxy: dict[str, str] = {"server": server}
    if parsed.username:
        proxy["username"] = parsed.username
    if parsed.password:
        proxy["password"] = parsed.password
    return proxy


class BrowserFetcher:
    """Playwright headless-Chromium fetcher with the ``ScraperClient`` interface.

    Args:
        proxy: Proxy URL (``scheme://[user:pass@]host:port``). Overrides the
            ``UFC_SCRAPE_PROXY`` env var. A residential proxy is the documented
            mitigation for IP-based re-blocks.
        delay: Minimum seconds between consecutive requests (default 1.5).
        max_retries: Retries on a detected challenge before HALTing (default 3).
        timeout: Per-navigation timeout in seconds (default 45.0).
        wait_selector: Overrides the URL-aware default settling selector for
            every ``get`` call. Leave as the default to let each call pick a
            selector based on the URL (events table for the listing page, none
            for detail pages); pass ``None`` to disable the selector wait.
        headless: Run Chromium headless (default True).

    Notes:
        The browser + context + page are launched lazily on the first
        :meth:`get` and reused for the fetcher's lifetime, so the anti-bot
        challenge is solved once and its cookies persist across calls. Always
        call :meth:`close` (or use as a context manager) to release the browser.
    """

    def __init__(
        self,
        proxy: str | None = None,
        delay: float = 1.5,
        max_retries: int = 3,
        timeout: float = 45.0,
        wait_selector: str | None | object = _UNSET,
        headless: bool = True,
    ) -> None:
        self._proxy_url = proxy if proxy is not None else os.environ.get("UFC_SCRAPE_PROXY")
        self._delay = delay
        self._max_retries = max_retries
        self._timeout_ms = int(timeout * 1000)
        # _UNSET => derive the selector from each URL; anything else pins it.
        self._wait_selector_override = wait_selector
        self._headless = headless

        # Playwright handles, lazily populated by :meth:`_ensure_browser`.
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None

        self._last_request_time = 0.0

    # ── Lifecycle ────────────────────────────────────────────────────────

    def _launch(self) -> None:
        """Launch Chromium and open a stealth-configured page (solve once).

        Isolated so unit tests can patch it (or set ``self._page`` directly)
        without a real browser.
        """
        from playwright.sync_api import sync_playwright  # lazy: heavy import

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self._headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        self._context = self._browser.new_context(
            user_agent=_USER_AGENT,
            viewport=cast("ViewportSize", _VIEWPORT),
            locale="en-US",
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "image/avif,image/webp,*/*;q=0.8"
                ),
            },
            proxy=cast("ProxySettings | None", _parse_proxy(self._proxy_url)),
        )
        self._context.add_init_script(_STEALTH_INIT_SCRIPT)
        self._page = self._context.new_page()

    def _ensure_browser(self) -> None:
        """Launch the browser on first use; reuse the live context afterwards."""
        if self._page is None:
            self._launch()

    # ── Rate limiting ────────────────────────────────────────────────────

    def _respect_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)

    def _backoff(self, attempt: int) -> None:
        """Exponential backoff between challenge retries (5s, 10s, 20s)."""
        time.sleep(2**attempt * 5)

    # ── Fetch ────────────────────────────────────────────────────────────

    def get(self, url: str, wait_selector: str | None | object = _UNSET) -> str:
        """Navigate to ``url``, wait out the challenge, return rendered HTML.

        Args:
            url: The URL to fetch.
            wait_selector: CSS selector proving the real page rendered. Leave
                unset to derive it from the URL (events table for the listing
                page, ``None`` for detail pages); pass ``None`` to skip the
                wait. When a selector IS in effect and never appears, the page
                is treated as still-challenged (contributes to the HALT
                decision alongside :func:`detect_antibot` HTML signatures).
                Detail-page URLs derive ``None`` and so are never selector-gated
                — they rely on the already-solved Cloudflare clearance cookie.

        Returns:
            The fully rendered ``page.content()`` HTML on success.

        Raises:
            AntiBotChallengeError: If the challenge persists after all retries.
        """
        # Resolve the effective selector: explicit per-call arg > constructor
        # override > URL-derived default.
        if wait_selector is _UNSET:
            wait_selector = (
                self._wait_selector_override
                if self._wait_selector_override is not _UNSET
                else _default_selector_for(url)
            )

        self._ensure_browser()

        last_status = 0
        for attempt in range(self._max_retries + 1):
            self._respect_rate_limit()
            try:
                response = self._page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=self._timeout_ms,
                )
            finally:
                self._last_request_time = time.monotonic()

            status = getattr(response, "status", 200) if response is not None else 200
            last_status = status

            # Give the challenge JS time to run and the real content to appear.
            # The networkidle wait lets a client-side PoW redirect complete.
            try:
                self._page.wait_for_load_state("networkidle", timeout=self._timeout_ms)
            except Exception as exc:
                logger.debug("networkidle wait timed out on %s: %s", url, exc)

            selector_ok = True
            if wait_selector:
                try:
                    self._page.wait_for_selector(wait_selector, timeout=self._timeout_ms)
                except Exception as exc:
                    selector_ok = False
                    logger.debug("selector %s not found on %s: %s", wait_selector, url, exc)

            html: str = self._page.content()

            # Block decision. NOTE: after the browser solves a JS proof-of-work
            # challenge, ``response.status`` still reflects the INITIAL 403
            # challenge response (the solve happens client-side, not via an
            # HTTP redirect), so the raw status is unreliable here. We therefore
            # decide on the RENDERED HTML: challenge signatures present, or a
            # required content selector that never rendered. ``status`` is only
            # logged for diagnostics.
            challenged = detect_antibot(html, 200)  # signatures only, ignore stale status
            content_missing = bool(wait_selector) and not selector_ok
            if not challenged and not content_missing:
                logger.info(
                    "browser fetch OK: %s (http_status=%s, %d bytes, attempt %d)",
                    url,
                    status,
                    len(html),
                    attempt + 1,
                )
                return html

            logger.warning(
                "anti-bot challenge on %s (http_status=%s, challenged=%s, "
                "content_missing=%s), attempt %d/%d",
                url,
                status,
                challenged,
                content_missing,
                attempt + 1,
                self._max_retries,
            )
            if attempt < self._max_retries:
                self._backoff(attempt)

        msg = (
            f"UFCStats anti-bot challenge persisted for {url} after "
            f"{self._max_retries} retries (last status={last_status}). "
            "HALTing honestly rather than returning challenge/stub data. "
            "A residential proxy (UFC_SCRAPE_PROXY) may be required."
        )
        raise AntiBotChallengeError(msg)

    # ── Batch (ScraperClient-compatible, serial / order-preserving) ──────

    def map(self, fn: Callable[[str], T], urls: Sequence[str]) -> list[T]:
        """Apply ``fn`` to each URL serially, preserving input order.

        Mirrors ``ScraperClient.map`` with ``workers=1``: a single browser
        context means fetches are inherently serial. If ``fn`` raises, the
        exception propagates (first failure wins).
        """
        return [fn(url) for url in urls]

    def map_get(self, urls: Sequence[str]) -> list[str]:
        """Fetch all ``urls`` via :meth:`get`, preserving input order."""
        return self.map(self.get, urls)

    # ── Teardown ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the page, context, browser, and Playwright driver."""
        for name, obj in (
            ("page", self._page),
            ("context", self._context),
            ("browser", self._browser),
        ):
            if obj is not None:
                try:
                    obj.close()
                except Exception as exc:
                    logger.debug("error closing browser %s: %s", name, exc)
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception as exc:
                logger.debug("error stopping playwright: %s", exc)
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None

    def __enter__(self) -> BrowserFetcher:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
