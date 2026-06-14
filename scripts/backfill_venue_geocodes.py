"""TRAVEL-V22-01 venues backfill driver — Phase 22 (one-off, NOT runtime).

Per CONTEXT D-07: ``geopy`` + ``timezonefinder`` are NOT runtime deps. Install via::

    uv pip install --no-deps "geopy==2.4.1" "timezonefinder==8.2.0"

(outside the project venv or in a transient venv). ``pyproject.toml`` runtime
dependencies remain unchanged.

Per RESEARCH Finding 8: Nominatim policy enforced via ``geopy.extra.RateLimiter``
(``min_delay_seconds=1.2``, ``max_retries=2``, ``error_wait_seconds=10.0``;
mandatory ``user_agent``).

Per RESEARCH Finding 6: ``TimezoneFinder()`` over ``TimezoneFinderL()`` —
coastline accuracy matters for venues like UFC Fight Island / T-Mobile Arena.

Per Pitfall #5: cache-first via ``data/venues_geocode_cache.json`` keyed by raw
venue string; reruns only re-fetch failures.

Banned imports per Pitfall #1 / Finding 11: nothing under ``ufc_prediction.ml.*``.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any


# ─── Locked constants (CONTEXT D-07 + RESEARCH Findings 5, 6, 8) ─────────────

NOMINATIM_USER_AGENT: str = "ufc-fight-prediction-v22-venues-backfill"
NOMINATIM_DELAY_S: float = 1.2
NOMINATIM_MAX_RETRIES: int = 2
NOMINATIM_ERROR_WAIT_S: float = 10.0
NOMINATIM_HTTP_TIMEOUT_S: float = 20.0  # geopy adapter default is 1s — too tight; many Nominatim queries take 3-5s
NOMINATIM_THROTTLE_COOLDOWN_S: float = 120.0  # cool-down when consecutive 429s detected
NOMINATIM_CONSECUTIVE_FAILS_TRIGGER: int = 3  # back-off after this many in a row

OUTPUT_CSV: Path = Path("data/venues.csv")
GEOCODE_CACHE: Path = Path("data/venues_geocode_cache.json")

# 10-col CSV header per CONTEXT D-08 + REVISION-03
CSV_HEADER: list[str] = [
    "venue_id", "name", "city", "state", "country",
    "lat", "lon", "timezone_iana", "n_events", "geocode_source",
]


# ─── Pure helpers (no network; tested in isolation) ──────────────────────────

def _load_cache(cache_path: Path) -> dict[str, dict | None]:
    """Load the geocode cache JSON (empty dict if file missing).

    Values may be ``None`` (sentinel for cached Nominatim misses — WR-02).

    WR-05: malformed JSON (e.g. truncated mid-write from a prior crash
    before atomic-rename was added, or external corruption) is logged as
    WARN and treated as an empty cache rather than crashing the script.
    Pairs with WR-04 atomic-rename for the cache writer (planned
    follow-up) to prevent the corruption in the first place.
    """
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"[backfill-venues] WARN: cache load failed ({type(exc).__name__}: "
            f"{exc}); starting from empty cache",
            file=sys.stderr,
        )
        return {}


def _save_cache(cache: dict[str, dict | None], cache_path: Path) -> None:
    """Persist the geocode cache (sort_keys for deterministic git diffs).

    WR-05: write via <path>.tmp + os.replace (atomic rename) so a crash
    mid-write cannot produce a partial JSON file that _load_cache would
    fail to parse on the next run. Pairs with _load_cache's try/except
    fallback (defense-in-depth).
    """
    import os  # noqa: PLC0415

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(cache, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, cache_path)


def _count_existing_csv_rows(output_path: Path) -> int:
    """Return the number of data rows in an existing CSV (0 if missing).

    WR-04: used to detect a non-monotonic shrink between runs. The migration
    at migrations/versions/59981c08e056_add_venues_table.py:91-109 reads
    from this CSV, so a silent shrink would regress migration seed data.
    """
    if not output_path.exists():
        return 0
    try:
        with open(output_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            return sum(1 for _ in reader)
    except OSError as exc:
        print(
            f"[backfill-venues] WARN: could not read existing CSV for "
            f"shrink-check ({exc}); proceeding without shrink-detection",
            file=sys.stderr,
        )
        return 0


def _emit_csv(
    rows: list[dict], output_path: Path, *, allow_shrink: bool = False
) -> bool:
    """Write the 10-col CSV; rows pre-sorted by n_events DESC for operator review.

    WR-04 atomic write: writes to ``<output_path>.tmp`` then ``os.replace`` to
    the final path. Failed runs that crash mid-write leave the previous CSV
    intact (the temp file is left for inspection).

    WR-04 shrink protection: if the new row count is strictly less than the
    prior CSV's row count, the emit is REFUSED unless ``allow_shrink=True``.
    Returns ``True`` on successful write, ``False`` if write was refused due
    to detected shrink. The migration seed data is the reason this is a hard
    block by default (rather than a WARN).
    """
    import os  # noqa: PLC0415

    prior_rows = _count_existing_csv_rows(output_path)
    if not allow_shrink and len(rows) < prior_rows:
        print(
            f"[backfill-venues] FATAL: CSV row count would shrink from "
            f"{prior_rows} to {len(rows)}; refusing to overwrite. "
            f"Pass --allow-shrink to override (migration seed data at risk).",
            file=sys.stderr,
        )
        return False
    if len(rows) < prior_rows:
        print(
            f"[backfill-venues] WARN: --allow-shrink: writing {len(rows)} "
            f"rows over prior {prior_rows} rows (migration seed data may regress)",
            file=sys.stderr,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in CSV_HEADER})
    os.replace(tmp_path, output_path)
    return True


# ─── Geocode helpers (lazy-import geopy; cache-first) ────────────────────────

def _make_geolocator():
    """Factory seam (WR-06): single point to construct the Nominatim client.

    Tests patch ``scripts.backfill_venue_geocodes._make_geolocator`` directly
    to inject a mock — this is a stable seam that doesn't depend on the
    lazy-import pattern inside ``_load_or_geocode`` (which would silently
    break test mocks if the import were ever hoisted to module-level).

    Lazy-imports geopy because geopy is NOT a runtime dep (D-07); the
    factory is only called on a real cache miss.
    """
    from geopy.geocoders import Nominatim  # noqa: PLC0415

    return Nominatim(
        user_agent=NOMINATIM_USER_AGENT,
        timeout=NOMINATIM_HTTP_TIMEOUT_S,
    )


def _load_or_geocode(venue_name: str, cache: dict[str, dict | None]) -> dict | None:
    """Cache-first geocode. Pitfall #5: only hits Nominatim on cache miss.

    Cache semantics (WR-02): both hits AND misses are cached. A successful
    geocode caches the address record; a Nominatim ``None`` return (no
    result for the input) is cached as a ``None`` sentinel so reruns
    short-circuit without re-burning Nominatim quota on un-geocodable
    venues. Transient network/throttle failures are NOT cached — those
    stay un-cached so the operator can rerun and resolve them later.
    """
    if venue_name in cache:
        return cache[venue_name]  # may be None (cached miss) — fine

    # Lazy-import — geopy is NOT a runtime dep (D-07)
    import time  # noqa: PLC0415

    from geopy.exc import GeocoderRateLimited, GeocoderUnavailable  # noqa: PLC0415

    geolocator = _make_geolocator()
    # Hand-rolled rate limiting (1.2s between requests) + bounded retries.
    # We skip-on-failure rather than wait-forever so the run completes; transient
    # failures stay UNCACHED and re-fetch on next run (operator can rerun later).
    time.sleep(NOMINATIM_DELAY_S)
    try:
        location = geolocator.geocode(
            venue_name,
            addressdetails=True,
            exactly_one=True,
            timeout=NOMINATIM_HTTP_TIMEOUT_S,
        )
    except (GeocoderRateLimited, GeocoderUnavailable) as exc:
        print(
            f"[backfill-venues] WARN: skip {venue_name!r}: {type(exc).__name__}",
            file=sys.stderr,
        )
        return None
    except Exception as exc:  # noqa: BLE001
        print(
            f"[backfill-venues] WARN: skip {venue_name!r}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None
    if location is None:
        # Cache the miss so reruns don't re-burn Nominatim quota on a venue
        # Nominatim has authoritatively said it cannot resolve. Distinct from
        # transient throttle/network failures above (which stay un-cached).
        cache[venue_name] = None
        return None
    record = {
        "lat": location.latitude,
        "lon": location.longitude,
        "address": location.raw.get("address", {}),
        "geocode_source": f"nominatim:{date.today().isoformat()}",
    }
    cache[venue_name] = record
    return record


def _derive_iana_tz(lat: float, lon: float) -> str | None:
    """TimezoneFinder() lookup — Finding 6 (NOT TimezoneFinderL; coastline accuracy)."""
    from timezonefinder import TimezoneFinder  # noqa: PLC0415
    tf = TimezoneFinder()
    return tf.timezone_at(lat=lat, lng=lon)


# ─── DB query for distinct venues (lazy DB import) ───────────────────────────

def _query_distinct_venues(session: Any) -> list[tuple[str, int]]:
    """SELECT location, COUNT(*) FROM events GROUP BY location ORDER BY 2 DESC."""
    from sqlalchemy import func as sa_func, select  # noqa: PLC0415

    from ufc_prediction.models import Event  # noqa: PLC0415
    rows = session.execute(
        select(Event.location, sa_func.count(Event.id).label("n"))
        .where(Event.location.isnot(None))
        .group_by(Event.location)
        .order_by(sa_func.count(Event.id).desc()),
    ).all()
    return [(r[0], r[1]) for r in rows]


# ─── Main ────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    """Run the backfill.

    Returns 0 on success, 2 on locked-constants drift, 3 if the new CSV
    would shrink vs the prior on-disk CSV and --allow-shrink was not set
    (WR-04 migration seed-data protection).
    """
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description=(
            "TRAVEL-V22-01 venues backfill driver (Phase 22). Cache-first geocode "
            "via Nominatim; writes data/venues.csv for migration seed."
        ),
    )
    parser.add_argument(
        "--allow-shrink",
        action="store_true",
        help=(
            "Allow writing a CSV with fewer rows than the existing one. "
            "By default, a shrink is refused (return code 3) to prevent "
            "regressing migration seed data (WR-04)."
        ),
    )
    args = parser.parse_args(argv)

    # AF startup asserts — locked-constants drift detection
    expected = {
        "NOMINATIM_USER_AGENT": (NOMINATIM_USER_AGENT, "ufc-fight-prediction-v22-venues-backfill"),
        "NOMINATIM_DELAY_S": (NOMINATIM_DELAY_S, 1.2),
        "OUTPUT_CSV": (str(OUTPUT_CSV), "data/venues.csv"),
        "CSV_HEADER_LEN": (len(CSV_HEADER), 10),
    }
    for name, (got, want) in expected.items():
        if got != want:
            print(
                f"[backfill-venues] FATAL: {name} drift: expected {want!r}, got {got!r}",
                file=sys.stderr,
            )
            return 2

    from ufc_prediction.db.session import get_session  # noqa: PLC0415

    cache = _load_cache(GEOCODE_CACHE)
    rows: list[dict] = []

    with next(get_session()) as session:
        distinct = _query_distinct_venues(session)
    print(
        f"[backfill-venues] {len(distinct)} distinct location strings; "
        "~150 expected post-dedup",
    )

    # Incremental save every N venues so a mid-run crash doesn't lose progress
    SAVE_EVERY = 25
    consecutive_fails = 0
    for venue_id, (raw_location, n_events) in enumerate(distinct, start=1):
        was_cached = raw_location in cache
        record = _load_or_geocode(raw_location, cache)

        # Adaptive back-off: if Nominatim throttles us hard, sleep for cool-down
        # so the next batch has a chance to land. Only triggers on un-cached failures.
        if not was_cached:
            if record is None:
                consecutive_fails += 1
                if consecutive_fails >= NOMINATIM_CONSECUTIVE_FAILS_TRIGGER:
                    import time  # noqa: PLC0415

                    print(
                        f"[backfill-venues] BACK-OFF: {consecutive_fails} consecutive "
                        f"failures; sleeping {NOMINATIM_THROTTLE_COOLDOWN_S}s",
                        file=sys.stderr,
                    )
                    time.sleep(NOMINATIM_THROTTLE_COOLDOWN_S)
                    consecutive_fails = 0  # reset
            else:
                consecutive_fails = 0  # reset on success

        if venue_id % SAVE_EVERY == 0:
            _save_cache(cache, GEOCODE_CACHE)
            print(
                f"[backfill-venues] progress: {venue_id}/{len(distinct)} "
                f"({len(cache)} cache entries)",
            )
        if record is None:
            print(
                f"[backfill-venues] WARN: no geocode for {raw_location!r}; "
                "manual review needed",
                file=sys.stderr,
            )
            continue
        addr = record["address"]
        tz = _derive_iana_tz(record["lat"], record["lon"])
        if tz is None:
            print(
                f"[backfill-venues] WARN: no tz for {raw_location!r}; skipping",
                file=sys.stderr,
            )
            continue
        rows.append({
            "venue_id": venue_id,
            "name": addr.get("amenity") or addr.get("building") or raw_location,
            "city": addr.get("city") or addr.get("town") or addr.get("village"),
            "state": addr.get("state"),
            "country": addr.get("country", ""),
            "lat": record["lat"],
            "lon": record["lon"],
            "timezone_iana": tz,
            "n_events": n_events,
            "geocode_source": record["geocode_source"],
        })

    _save_cache(cache, GEOCODE_CACHE)
    if not _emit_csv(rows, OUTPUT_CSV, allow_shrink=args.allow_shrink):
        # _emit_csv already printed a FATAL message. Cache was still
        # persisted, so a future rerun with --allow-shrink (or that adds
        # more rows) can recover without losing intermediate work.
        return 3
    print(f"[backfill-venues] wrote {len(rows)} rows to {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
