"""HTTP client wrapper for UFCStats.com scraping.

Provides rate limiting, retry with exponential backoff, and a custom
User-Agent header per project decisions D-01, D-02, D-03.

Supports optional concurrent scraping via a worker pool. When ``workers > 1``,
each worker thread enforces its own rate limit independently (via
``threading.local()``) so effective throughput is roughly ``workers / delay``
requests per second while remaining polite from any single worker's
perspective. ``httpx.Client`` is thread-safe for concurrent GET requests, so
a single underlying client is shared across workers.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

import httpx

logger = logging.getLogger(__name__)

_USER_AGENT = "UFCFightPrediction/0.1 (research project)"

T = TypeVar("T")


class ScraperClient:
    """HTTP client with rate limiting and retry for polite web scraping.

    Args:
        delay: Minimum seconds between consecutive requests from a single
            worker (default 1.5). With ``workers > 1`` each worker enforces
            this delay independently.
        max_retries: Maximum retry attempts on 429/503 (default 3).
        timeout: HTTP request timeout in seconds (default 30.0).
        workers: Maximum number of concurrent worker threads available to
            :meth:`map` / :meth:`map_get` (default 1). Must be >= 1. With
            ``workers=1`` (the default) behavior is identical to the
            pre-concurrency implementation: no threads are spawned and
            :meth:`map` runs inline.

    Notes:
        ``httpx.Client`` is thread-safe for concurrent GET requests, so one
        client instance is shared across workers. Rate limiting state lives
        in a ``threading.local()``, which means each worker thread independently
        waits out ``delay`` since its own last request. Effective aggregate
        throughput is therefore approximately ``workers / delay`` req/s.
    """

    def __init__(
        self,
        delay: float = 1.5,
        max_retries: int = 3,
        timeout: float = 30.0,
        workers: int = 1,
    ) -> None:
        if workers < 1:
            msg = f"workers must be >= 1, got {workers}"
            raise ValueError(msg)

        self._delay = delay
        self._max_retries = max_retries
        self._workers = workers
        self._client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
        )
        # Per-thread rate limit state. Each worker thread sees its own
        # ``last_request_time`` attribute (0.0 before the first request).
        self._tls = threading.local()
        # Lazily created on first map() call when workers > 1.
        self._executor: ThreadPoolExecutor | None = None

    def get(self, url: str) -> str:
        """Fetch URL with rate limiting and retry.

        Rate limits: sleeps if elapsed time since this thread's last request
        < ``delay``. Retries on 429/503 with exponential backoff (5s, 10s, 20s).
        Raises on non-200 after retries exhausted.

        Args:
            url: The URL to fetch.

        Returns:
            The response body text on HTTP 200.

        Raises:
            RuntimeError: If all retry attempts are exhausted (429/503).
            httpx.HTTPStatusError: On non-retryable HTTP errors.
        """
        # Rate limit (per-thread state)
        last_request_time = getattr(self._tls, "last_request_time", 0.0)
        elapsed = time.monotonic() - last_request_time
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.get(url)
            except (httpx.TransportError, ConnectionError, OSError) as exc:
                # Transport-level failure: TCP reset, read timeout, pool closed,
                # RemoteProtocolError (server closed connection on keep-alive
                # limit), etc. UFCStats resets connections after N requests on
                # the same keep-alive; httpx surfaces this as a TransportError
                # rather than an HTTP status code.
                self._tls.last_request_time = time.monotonic()
                if attempt < self._max_retries:
                    backoff = 2**attempt * 5  # 5s, 10s, 20s
                    logger.warning(
                        "Transport error on %s (%s: %s), retrying in %ds (attempt %d/%d)",
                        url,
                        type(exc).__name__,
                        exc,
                        backoff,
                        attempt + 1,
                        self._max_retries,
                    )
                    time.sleep(backoff)
                    continue
                msg = (
                    f"Failed to fetch {url} after {self._max_retries} retries: "
                    f"{type(exc).__name__}: {exc}"
                )
                raise RuntimeError(msg) from exc

            self._tls.last_request_time = time.monotonic()

            if response.status_code == 200:
                return response.text

            if response.status_code in (429, 503) and attempt < self._max_retries:
                backoff = 2**attempt * 5  # 5s, 10s, 20s
                logger.warning(
                    "Got %d from %s, retrying in %ds (attempt %d/%d)",
                    response.status_code,
                    url,
                    backoff,
                    attempt + 1,
                    self._max_retries,
                )
                time.sleep(backoff)
                continue

            # Non-retryable or last retry attempt for retryable codes
            if response.status_code not in (429, 503):
                response.raise_for_status()

        msg = f"Failed to fetch {url} after {self._max_retries} retries"
        raise RuntimeError(msg)

    def map(
        self,
        fn: Callable[[str], T],
        urls: Sequence[str],
    ) -> list[T]:
        """Apply ``fn`` to each URL, optionally in parallel.

        With ``workers == 1`` (the default) this runs inline — no pool is
        created — so behavior is byte-for-byte identical to a serial loop of
        ``fn(url) for url in urls``.

        With ``workers > 1`` the first call lazily creates a
        :class:`concurrent.futures.ThreadPoolExecutor` with ``max_workers =
        workers`` and dispatches ``fn`` across it. Each worker thread enforces
        its own rate limit, so total wall-clock is roughly
        ``len(urls) * delay / workers`` (plus the cost of ``fn`` itself).

        Results are always returned in the SAME order as ``urls``. If ``fn``
        raises for any URL the exception propagates out of :meth:`map`
        (first failure wins).

        Args:
            fn: Callable applied to each URL. Most commonly :meth:`get`.
            urls: URLs to process.

        Returns:
            A list of results, one per URL, in input order.

        Raises:
            Exception: Whatever ``fn`` raises for the first failing URL.
        """
        if self._workers == 1:
            return [fn(url) for url in urls]

        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._workers,
                thread_name_prefix="scraper",
            )
        return list(self._executor.map(fn, urls))

    def map_get(self, urls: Sequence[str]) -> list[str]:
        """Fetch all ``urls`` via :meth:`get`, preserving input order.

        Convenience wrapper over ``self.map(self.get, urls)``.
        """
        return self.map(self.get, urls)

    def close(self) -> None:
        """Close the underlying HTTP client and any worker pool."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        self._client.close()

    def __enter__(self) -> ScraperClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
