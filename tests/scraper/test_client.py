"""Unit tests for ScraperClient HTTP wrapper.

Tests cover rate limiting, retry with exponential backoff, User-Agent header,
error handling for non-retryable status codes, and concurrent dispatch via
the ``workers`` / ``map`` / ``map_get`` API.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import httpx
import pytest

from ufc_prediction.scraper.client import ScraperClient


@pytest.fixture
def mock_response_200():
    """Create a mock 200 response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.text = "<html>OK</html>"
    return resp


@pytest.fixture
def mock_response_429():
    """Create a mock 429 response."""
    resp = MagicMock()
    resp.status_code = 429
    resp.text = "Too Many Requests"
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "429 Too Many Requests",
        request=MagicMock(),
        response=resp,
    )
    return resp


@pytest.fixture
def mock_response_503():
    """Create a mock 503 response."""
    resp = MagicMock()
    resp.status_code = 503
    resp.text = "Service Unavailable"
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "503 Service Unavailable",
        request=MagicMock(),
        response=resp,
    )
    return resp


@pytest.fixture
def mock_response_522():
    """Create a mock 522 response (Cloudflare 'Connection Timed Out')."""
    resp = MagicMock()
    resp.status_code = 522
    resp.text = "Connection timed out"
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "522 <none>",
        request=MagicMock(),
        response=resp,
    )
    return resp


@pytest.fixture
def mock_response_404():
    """Create a mock 404 response."""
    resp = MagicMock()
    resp.status_code = 404
    resp.text = "Not Found"
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "404 Not Found",
        request=MagicMock(),
        response=resp,
    )
    return resp


class TestScraperClientGet:
    """Tests for ScraperClient.get() method."""

    @patch("ufc_prediction.scraper.client.time")
    @patch("ufc_prediction.scraper.client.httpx.Client")
    def test_get_returns_text_on_200(self, mock_client_cls, mock_time, mock_response_200):
        """get() returns response.text on HTTP 200."""
        mock_time.monotonic.return_value = 100.0
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response_200
        mock_client_cls.return_value = mock_client

        client = ScraperClient(delay=0.0)
        result = client.get("http://example.com")

        assert result == "<html>OK</html>"
        mock_client.get.assert_called_once_with("http://example.com")

    @patch("ufc_prediction.scraper.client.time")
    @patch("ufc_prediction.scraper.client.httpx.Client")
    def test_get_sets_user_agent(self, mock_client_cls, mock_time):
        """ScraperClient sets the correct User-Agent header."""
        mock_time.monotonic.return_value = 100.0
        ScraperClient(delay=0.0)

        # Verify the httpx.Client was created with the correct User-Agent
        call_kwargs = mock_client_cls.call_args[1]
        assert call_kwargs["headers"]["User-Agent"] == "UFCFightPrediction/0.1 (research project)"

    @patch("ufc_prediction.scraper.client.time")
    @patch("ufc_prediction.scraper.client.httpx.Client")
    def test_get_rate_limits(self, mock_client_cls, mock_time, mock_response_200):
        """get() sleeps to enforce rate limiting between consecutive calls."""
        # First call at t=100.0, second call check at t=100.3 (only 0.3s elapsed)
        mock_time.monotonic.side_effect = [0.0, 100.0, 100.0, 100.3, 100.3]
        mock_time.sleep = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response_200
        mock_client_cls.return_value = mock_client

        client = ScraperClient(delay=1.0)
        client.get("http://example.com/page1")
        client.get("http://example.com/page2")

        # Should have slept for the remaining delay
        mock_time.sleep.assert_called()
        sleep_arg = mock_time.sleep.call_args[0][0]
        assert sleep_arg > 0, "Should sleep to enforce rate limit"

    @patch("ufc_prediction.scraper.client.time")
    @patch("ufc_prediction.scraper.client.httpx.Client")
    def test_get_retries_on_429(
        self, mock_client_cls, mock_time, mock_response_429, mock_response_200
    ):
        """get() retries on 429 and succeeds on second attempt."""
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()
        mock_client = MagicMock()
        mock_client.get.side_effect = [mock_response_429, mock_response_200]
        mock_client_cls.return_value = mock_client

        client = ScraperClient(delay=0.0, max_retries=3)
        result = client.get("http://example.com")

        assert result == "<html>OK</html>"
        assert mock_client.get.call_count == 2

    @patch("ufc_prediction.scraper.client.time")
    @patch("ufc_prediction.scraper.client.httpx.Client")
    def test_get_retries_on_503(
        self, mock_client_cls, mock_time, mock_response_503, mock_response_200
    ):
        """get() retries on 503 and succeeds on second attempt."""
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()
        mock_client = MagicMock()
        mock_client.get.side_effect = [mock_response_503, mock_response_200]
        mock_client_cls.return_value = mock_client

        client = ScraperClient(delay=0.0, max_retries=3)
        result = client.get("http://example.com")

        assert result == "<html>OK</html>"
        assert mock_client.get.call_count == 2

    @patch("ufc_prediction.scraper.client.time")
    @patch("ufc_prediction.scraper.client.httpx.Client")
    def test_get_retries_on_522(
        self, mock_client_cls, mock_time, mock_response_522, mock_response_200
    ):
        """get() retries on Cloudflare 522 and succeeds on second attempt.

        Regression guard for the weekly BFO refresh failure of 2026-07-01: a
        single transient 522 ("Connection Timed Out") from BestFightOdds behind
        Cloudflare aborted the whole scrape because 52x codes were not retryable.
        """
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()
        mock_client = MagicMock()
        mock_client.get.side_effect = [mock_response_522, mock_response_200]
        mock_client_cls.return_value = mock_client

        client = ScraperClient(delay=0.0, max_retries=3)
        result = client.get("http://example.com")

        assert result == "<html>OK</html>"
        assert mock_client.get.call_count == 2

    @pytest.mark.parametrize("status", [520, 521, 522, 523, 524])
    @patch("ufc_prediction.scraper.client.time")
    @patch("ufc_prediction.scraper.client.httpx.Client")
    def test_get_retries_on_cloudflare_5xx_family(
        self, mock_client_cls, mock_time, status, mock_response_200
    ):
        """get() retries the WHOLE Cloudflare origin-error family (520-524), not just 522.

        Guards against a refactor to a numeric predicate (e.g. range(520, 524),
        an easy off-by-one that would silently stop retrying 524) — review
        finding #9.
        """
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()
        resp = MagicMock()
        resp.status_code = status
        resp.text = "cloudflare origin error"
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"{status} <none>", request=MagicMock(), response=resp
        )
        mock_client = MagicMock()
        mock_client.get.side_effect = [resp, mock_response_200]
        mock_client_cls.return_value = mock_client

        client = ScraperClient(delay=0.0, max_retries=3)
        result = client.get("http://example.com")

        assert result == "<html>OK</html>"
        assert mock_client.get.call_count == 2

    def test_retryable_status_set_covers_expected_codes(self):
        """The retryable set must include 429/503 + the Cloudflare 520-524 family."""
        from ufc_prediction.scraper.client import _RETRYABLE_STATUS

        assert {429, 503, 520, 521, 522, 523, 524} <= _RETRYABLE_STATUS

    @patch("ufc_prediction.scraper.client.time")
    @patch("ufc_prediction.scraper.client.httpx.Client")
    def test_get_raises_after_max_retries(self, mock_client_cls, mock_time, mock_response_429):
        """get() raises RuntimeError after exhausting all retries."""
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()
        mock_client = MagicMock()
        # 3 retries + 1 initial = 4 attempts, all 429
        mock_client.get.return_value = mock_response_429
        mock_client_cls.return_value = mock_client

        client = ScraperClient(delay=0.0, max_retries=3)

        with pytest.raises(RuntimeError, match=r"Failed to fetch.*after 3 retries"):
            client.get("http://example.com")

        assert mock_client.get.call_count == 4  # initial + 3 retries

    @patch("ufc_prediction.scraper.client.time")
    @patch("ufc_prediction.scraper.client.httpx.Client")
    def test_get_raises_on_non_retryable(self, mock_client_cls, mock_time, mock_response_404):
        """get() raises HTTPStatusError immediately on non-retryable status codes."""
        mock_time.monotonic.return_value = 100.0
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response_404
        mock_client_cls.return_value = mock_client

        client = ScraperClient(delay=0.0, max_retries=3)

        with pytest.raises(httpx.HTTPStatusError):
            client.get("http://example.com/missing")

        assert mock_client.get.call_count == 1  # No retries

    @patch("ufc_prediction.scraper.client.time")
    @patch("ufc_prediction.scraper.client.httpx.Client")
    def test_get_retries_on_transport_error(self, mock_client_cls, mock_time, mock_response_200):
        """get() retries on TransportError (TCP reset / connection closed) and succeeds."""
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()
        mock_client = MagicMock()
        mock_client.get.side_effect = [
            httpx.ReadError("Connection reset by peer"),
            mock_response_200,
        ]
        mock_client_cls.return_value = mock_client

        client = ScraperClient(delay=0.0, max_retries=3)
        result = client.get("http://example.com")

        assert result == "<html>OK</html>"
        assert mock_client.get.call_count == 2  # one failure + one success

    @patch("ufc_prediction.scraper.client.time")
    @patch("ufc_prediction.scraper.client.httpx.Client")
    def test_get_raises_after_transport_error_retries_exhausted(self, mock_client_cls, mock_time):
        """get() raises RuntimeError after max_retries transport errors."""
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.ReadError("Connection reset by peer")
        mock_client_cls.return_value = mock_client

        client = ScraperClient(delay=0.0, max_retries=3)

        with pytest.raises(RuntimeError, match=r"Failed to fetch.*ReadError"):
            client.get("http://example.com")

        assert mock_client.get.call_count == 4  # initial + 3 retries

    @patch("ufc_prediction.scraper.client.time")
    @patch("ufc_prediction.scraper.client.httpx.Client")
    def test_get_retries_on_os_connection_reset(
        self, mock_client_cls, mock_time, mock_response_200
    ):
        """get() also retries on bare ConnectionResetError escaping httpx."""
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()
        mock_client = MagicMock()
        mock_client.get.side_effect = [
            ConnectionResetError(54, "Connection reset by peer"),
            mock_response_200,
        ]
        mock_client_cls.return_value = mock_client

        client = ScraperClient(delay=0.0, max_retries=3)
        result = client.get("http://example.com")

        assert result == "<html>OK</html>"
        assert mock_client.get.call_count == 2


class TestScraperClientContextManager:
    """Tests for context manager support."""

    @patch("ufc_prediction.scraper.client.time")
    @patch("ufc_prediction.scraper.client.httpx.Client")
    def test_context_manager(self, mock_client_cls, mock_time):
        """ScraperClient works as a context manager."""
        mock_time.monotonic.return_value = 0.0
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        with ScraperClient(delay=0.0) as client:
            assert client is not None

        mock_client.close.assert_called_once()


class TestScraperClientWorkersValidation:
    """Tests for the workers constructor parameter (Task 1 done criteria)."""

    @patch("ufc_prediction.scraper.client.httpx.Client")
    def test_workers_default_is_one(self, mock_client_cls):
        """Default construction preserves single-threaded behavior (workers=1)."""
        client = ScraperClient(delay=0.0)
        assert client._workers == 1

    @patch("ufc_prediction.scraper.client.httpx.Client")
    def test_workers_zero_raises_value_error(self, mock_client_cls):
        """workers=0 is invalid and must raise ValueError."""
        with pytest.raises(ValueError, match="workers"):
            ScraperClient(delay=0.0, workers=0)

    @patch("ufc_prediction.scraper.client.httpx.Client")
    def test_workers_negative_raises_value_error(self, mock_client_cls):
        """workers=-1 is invalid and must raise ValueError."""
        with pytest.raises(ValueError, match="workers"):
            ScraperClient(delay=0.0, workers=-1)

    @patch("ufc_prediction.scraper.client.httpx.Client")
    def test_workers_explicit_gt_one_accepted(self, mock_client_cls):
        """workers>1 is accepted and stored."""
        client = ScraperClient(delay=0.0, workers=4)
        assert client._workers == 4


def _make_instant_200_client():
    """Build a MagicMock httpx.Client whose .get returns an instant 200 response.

    Records the thread name on each call so tests can assert concurrency.
    """
    thread_names: list[str] = []
    lock = threading.Lock()

    def fake_get(url):
        with lock:
            thread_names.append(threading.current_thread().name)
        resp = MagicMock()
        resp.status_code = 200
        resp.text = f"<html>{url}</html>"
        return resp

    mock_client = MagicMock()
    mock_client.get.side_effect = fake_get
    mock_client.thread_names = thread_names  # expose for assertions
    return mock_client


class TestScraperClientConcurrency:
    """Tests for the workers / map / map_get concurrent-dispatch API."""

    @patch("ufc_prediction.scraper.client.httpx.Client")
    def test_map_preserves_order(self, mock_client_cls):
        """map() returns results in input order even with workers>1."""
        mock_client_cls.return_value = _make_instant_200_client()
        client = ScraperClient(delay=0.0, workers=4)

        results = client.map(lambda u: u, ["a", "b", "c", "d", "e", "f"])

        assert results == ["a", "b", "c", "d", "e", "f"]
        client.close()

    @patch("ufc_prediction.scraper.client.httpx.Client")
    def test_map_get_dispatches_concurrently(self, mock_client_cls):
        """With workers=4, 4 URLs at delay=0.3s complete well under serial 1.2s."""
        mock_client_cls.return_value = _make_instant_200_client()
        client = ScraperClient(delay=0.3, workers=4)

        start = time.monotonic()
        results = client.map_get(["u1", "u2", "u3", "u4"])
        elapsed = time.monotonic() - start

        # Serial equivalent would be roughly 4 * 0.3 = 1.2s (minus the first
        # call which has no prior timestamp). Parallel should be near-instant
        # because each worker sees a fresh thread-local last_request_time of
        # 0.0 on its first call.
        assert elapsed < 0.8, f"Expected concurrent execution, got {elapsed:.2f}s"
        assert len(results) == 4
        assert results[0] == "<html>u1</html>"
        client.close()

    @patch("ufc_prediction.scraper.client.httpx.Client")
    def test_per_worker_rate_limit_independent(self, mock_client_cls):
        """Each worker thread enforces its own delay (per-thread, not global)."""
        fake_client = _make_instant_200_client()
        mock_client_cls.return_value = fake_client
        client = ScraperClient(delay=0.1, workers=4)

        urls = [f"u{i}" for i in range(4)]
        client.map_get(urls)

        # At least 2 distinct worker threads must have serviced the 4 URLs;
        # this proves requests were not serialized through a single thread.
        distinct_threads = set(fake_client.thread_names)
        assert len(distinct_threads) >= 2, (
            f"Expected multi-worker dispatch, saw threads: {fake_client.thread_names}"
        )
        # All worker threads should have the "scraper" prefix configured on
        # the ThreadPoolExecutor.
        assert all(name.startswith("scraper") for name in fake_client.thread_names), (
            f"Unexpected thread names: {fake_client.thread_names}"
        )
        client.close()

    @patch("ufc_prediction.scraper.client.httpx.Client")
    def test_map_workers_one_runs_inline(self, mock_client_cls):
        """workers=1 takes the inline fast path: no ThreadPoolExecutor spawned."""
        mock_client_cls.return_value = _make_instant_200_client()
        client = ScraperClient(delay=0.0, workers=1)

        results = client.map(lambda u: u, ["a", "b"])

        assert results == ["a", "b"]
        assert client._executor is None, "workers=1 must not spawn a ThreadPoolExecutor"
        client.close()

    @patch("ufc_prediction.scraper.client.httpx.Client")
    def test_map_propagates_exception(self, mock_client_cls):
        """Exceptions raised by fn propagate out of map()."""
        mock_client_cls.return_value = _make_instant_200_client()
        client = ScraperClient(delay=0.0, workers=2)

        def boom(url: str) -> str:
            if url == "bad":
                msg = "oops"
                raise ValueError(msg)
            return url

        with pytest.raises(ValueError, match="oops"):
            client.map(boom, ["good", "bad", "good2"])
        client.close()

    @patch("ufc_prediction.scraper.client.httpx.Client")
    def test_map_propagates_exception_workers_one(self, mock_client_cls):
        """Inline path (workers=1) also propagates fn exceptions."""
        mock_client_cls.return_value = _make_instant_200_client()
        client = ScraperClient(delay=0.0, workers=1)

        def boom(url: str) -> str:
            if url == "bad":
                msg = "oops-serial"
                raise ValueError(msg)
            return url

        with pytest.raises(ValueError, match="oops-serial"):
            client.map(boom, ["good", "bad"])
        client.close()

    @patch.object(ThreadPoolExecutor, "shutdown", autospec=True)
    @patch("ufc_prediction.scraper.client.httpx.Client")
    def test_close_shuts_down_executor(self, mock_client_cls, mock_shutdown):
        """close() shuts down the executor if one was lazily created."""
        mock_client_cls.return_value = _make_instant_200_client()
        client = ScraperClient(delay=0.0, workers=4)
        client.map_get(["a", "b"])  # force executor creation
        assert client._executor is not None

        client.close()

        # The executor's shutdown was called at least once via close().
        assert mock_shutdown.called, "Expected close() to shut down the executor"

    @patch("ufc_prediction.scraper.client.httpx.Client")
    def test_close_noop_when_no_executor(self, mock_client_cls):
        """close() on an unused client (workers=1 or never mapped) is fine."""
        mock_client_cls.return_value = _make_instant_200_client()
        client = ScraperClient(delay=0.0, workers=4)
        # Never call map(); executor should be None.
        assert client._executor is None
        client.close()  # must not raise
