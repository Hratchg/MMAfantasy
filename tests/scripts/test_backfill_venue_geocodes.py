"""Wave-0 tests for scripts/backfill_venue_geocodes.py.

Pattern: ``importorskip`` for ``scripts.backfill_venue_geocodes`` + Mock Nominatim
calls. CRITICAL: zero live HTTP calls in this test suite (Nominatim policy
violation per RESEARCH Finding 8).

Banned imports per Pitfall #1 / Finding 11: nothing under ``ufc_prediction.ml.*``.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def backfill():
    """Import the backfill module (importorskip handles dev-only deps gracefully)."""
    return pytest.importorskip("scripts.backfill_venue_geocodes")


class TestLockedConstants:
    def test_user_agent_locked(self, backfill) -> None:
        assert backfill.NOMINATIM_USER_AGENT == "ufc-fight-prediction-v22-venues-backfill"

    def test_delay_locked(self, backfill) -> None:
        assert backfill.NOMINATIM_DELAY_S == 1.2

    def test_max_retries_locked(self, backfill) -> None:
        assert backfill.NOMINATIM_MAX_RETRIES == 2

    def test_error_wait_locked(self, backfill) -> None:
        assert backfill.NOMINATIM_ERROR_WAIT_S == 10.0

    def test_csv_header_10_cols(self, backfill) -> None:
        assert len(backfill.CSV_HEADER) == 10
        assert backfill.CSV_HEADER == [
            "venue_id", "name", "city", "state", "country",
            "lat", "lon", "timezone_iana", "n_events", "geocode_source",
        ]

    def test_output_path_committed_to_repo(self, backfill) -> None:
        assert str(backfill.OUTPUT_CSV) == "data/venues.csv"

    def test_cache_path_committed_to_repo(self, backfill) -> None:
        assert str(backfill.GEOCODE_CACHE) == "data/venues_geocode_cache.json"


class TestCacheFirstGeocode:
    def test_cache_hit_no_network(self, backfill) -> None:
        cached = {
            "lat": 36.1,
            "lon": -115.2,
            "address": {"city": "Las Vegas", "country": "USA"},
            "geocode_source": "nominatim:2026-05-15",
        }
        cache = {"T-Mobile Arena, Las Vegas, USA": cached}
        # If this calls Nominatim it would raise (no network mock) —
        # cache hit MUST short-circuit before any geopy import.
        result = backfill._load_or_geocode("T-Mobile Arena, Las Vegas, USA", cache)
        assert result == cached

    def test_cache_miss_calls_nominatim_and_stores(self, backfill) -> None:
        # importorskip geopy — if not installed, skip this test path
        pytest.importorskip("geopy")
        cache: dict = {}

        mock_loc = MagicMock(
            latitude=51.5,
            longitude=-0.12,
            raw={"address": {"city": "London", "country": "UK"}},
        )

        # WR-06: patch the factory seam at its lookup site in the script
        # module. This is durable against any future refactor that hoists
        # the Nominatim import to module-level, because _load_or_geocode
        # explicitly calls _make_geolocator() rather than constructing
        # Nominatim directly. Also patch time.sleep so the 1.2s
        # rate-limit doesn't slow the test.
        mock_geolocator = MagicMock()
        mock_geolocator.geocode.return_value = mock_loc
        with patch.object(backfill, "_make_geolocator", return_value=mock_geolocator), \
             patch("time.sleep"):
            backfill._load_or_geocode("O2 Arena, London", cache)

        assert "O2 Arena, London" in cache
        assert cache["O2 Arena, London"]["lat"] == 51.5
        assert cache["O2 Arena, London"]["lon"] == -0.12
        assert cache["O2 Arena, London"]["geocode_source"].startswith("nominatim:")

    def test_nominatim_none_return_is_cached_as_sentinel(self, backfill) -> None:
        """WR-02 regression: Nominatim None return is cached, second call no network."""
        pytest.importorskip("geopy")
        cache: dict = {}

        mock_geolocator = MagicMock()
        mock_geolocator.geocode.return_value = None  # Nominatim says "no result"
        with patch.object(backfill, "_make_geolocator", return_value=mock_geolocator), \
             patch("time.sleep"):
            first_result = backfill._load_or_geocode("Nonexistent Venue, XYZ", cache)
        assert first_result is None
        # The miss MUST be cached so reruns short-circuit
        assert "Nonexistent Venue, XYZ" in cache
        assert cache["Nonexistent Venue, XYZ"] is None

        # Second call: factory patch is gone — if _load_or_geocode tries to
        # hit Nominatim again, it would call the (now real, lazy-imported)
        # factory and crash without a network mock. Cache hit MUST prevent
        # that path entirely.
        second_result = backfill._load_or_geocode("Nonexistent Venue, XYZ", cache)
        assert second_result is None
        # Nominatim.geocode was called exactly once across both invocations
        assert mock_geolocator.geocode.call_count == 1


class TestEmitCsv:
    def test_emit_writes_10_col_header(self, backfill, tmp_path: Path) -> None:
        out = tmp_path / "venues.csv"
        backfill._emit_csv([], out)
        first_line = out.read_text(encoding="utf-8").splitlines()[0]
        assert first_line == (
            "venue_id,name,city,state,country,"
            "lat,lon,timezone_iana,n_events,geocode_source"
        )

    def test_emit_row_count(self, backfill, tmp_path: Path) -> None:
        out = tmp_path / "venues.csv"
        rows = [
            {
                "venue_id": 1,
                "name": "T-Mobile Arena",
                "city": "Las Vegas",
                "state": "Nevada",
                "country": "USA",
                "lat": 36.1,
                "lon": -115.2,
                "timezone_iana": "America/Los_Angeles",
                "n_events": 42,
                "geocode_source": "nominatim:2026-05-15",
            },
        ]
        backfill._emit_csv(rows, out)
        lines = out.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2  # header + 1 row


class TestCacheReproducibility:
    def test_rerun_produces_identical_cache(self, backfill, tmp_path: Path) -> None:
        cache = {
            f"Venue {i}": {
                "lat": 0.1 * i,
                "lon": 0.2 * i,
                "address": {"country": "X"},
                "geocode_source": "nominatim:2026-05-15",
            }
            for i in range(3)
        }
        cache_path = tmp_path / "cache.json"
        backfill._save_cache(cache, cache_path)
        loaded = backfill._load_cache(cache_path)
        assert loaded == cache

    def test_save_cache_sorted_for_deterministic_diff(self, backfill, tmp_path: Path) -> None:
        cache = {"Z venue": {"lat": 1}, "A venue": {"lat": 2}}
        cache_path = tmp_path / "cache.json"
        backfill._save_cache(cache, cache_path)
        text = cache_path.read_text(encoding="utf-8")
        # 'A venue' MUST appear before 'Z venue' (sort_keys=True)
        assert text.index("A venue") < text.index("Z venue")
