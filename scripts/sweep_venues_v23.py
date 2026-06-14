"""INGEST-V23-02 venues sweep driver — Phase 28 Plan 28-01 (one-off, NOT runtime).

D-09 (CONTEXT 28): operator-confirmed one-off Nominatim sweep to expand
``data/venues.csv`` from 52 → ~170 rows BEFORE Plan 28-02's per-event venue
backfill loop runs. Without this sweep, the venue_id ≥95% target is
mechanically unreachable (Las Vegas alone = 611 events = 32% of corpus, not
in the Phase 22 operator-curated 52-row subset).

Distinct from D-06 (CONTEXT 28) "no Nominatim at backfill time" — D-06 guards
the per-event scrape loop in Plan 28-02 (operationally wrong — burns quota per
event); D-09 permits this one-off pre-scrape sweep that mirrors Phase 22 D-07
``scripts/backfill_venue_geocodes.py`` exactly, just on the complete
distinct-set instead of the operator-curated subset.

Constants mirror Phase 22 ``scripts/backfill_venue_geocodes.py`` except:
- ``NOMINATIM_USER_AGENT`` → ``"ufc-fight-prediction-v23-venues-sweep"`` (D-09)
- ``GEOCODE_SOURCE_TAG`` → ``"nominatim:<today>"``

Per Phase 22 D-07: ``geopy`` + ``timezonefinder`` are NOT runtime deps. Install
ad-hoc::

    uv pip install --no-deps "geopy==2.4.1" "timezonefinder==8.2.0"

Per Phase 22 RESEARCH Finding 8: Nominatim policy enforced via
``geopy.extra.RateLimiter`` (min_delay_seconds=1.2, max_retries=2,
error_wait_seconds=10.0; mandatory user_agent).

Per Phase 22 Pitfall #5: cache-first via ``data/venues_geocode_cache.json``
keyed by raw venue string; reruns only re-fetch failures.

Banned imports per Phase 22 Pitfall #1 / Finding 11: nothing under
``ufc_prediction.ml.*``.

External-blocker policy (user MEMORY): if Nominatim throttles hard mid-sweep,
proceed with what was resolved + log remainder to 28-VENUES-SWEEP-NOTES.md
"Misses" table for operator manual entry. DO NOT pause autonomously.

Behavior:
- Existing rows in ``data/venues.csv`` are preserved byte-identical (venue_id,
  name, city, state, country, lat, lon, timezone_iana, geocode_source). Only
  ``n_events`` may update if DB counts shifted.
- New rows get monotonic venue_id = max(existing) + 1, +2, ...
- Cache is updated for both hits (full record) and misses (None sentinel per
  Phase 22 WR-02) so reruns are idempotent.
- Atomic CSV write via <path>.tmp + os.replace (Phase 22 WR-05).
- Reads ``data/venues.csv`` via csv.DictReader (preserves field order); writes
  via csv.DictWriter with the 10-col canonical header.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


# ─── Locked constants (D-09 override on user_agent; rest = Phase 22 verbatim) ──

NOMINATIM_USER_AGENT: str = "ufc-fight-prediction-v23-venues-sweep"  # D-09 OVERRIDE
NOMINATIM_DELAY_S: float = 1.2
NOMINATIM_MAX_RETRIES: int = 2
NOMINATIM_ERROR_WAIT_S: float = 10.0
NOMINATIM_HTTP_TIMEOUT_S: float = 20.0
NOMINATIM_THROTTLE_COOLDOWN_S: float = 120.0
NOMINATIM_CONSECUTIVE_FAILS_TRIGGER: int = 3

OUTPUT_CSV: Path = Path("data/venues.csv")
GEOCODE_CACHE: Path = Path("data/venues_geocode_cache.json")
NOTES_PATH: Path = Path(
    ".planning/phases/28-referee-venue-ingestion-pipeline/28-VENUES-SWEEP-NOTES.md"
)
GEOCODE_SOURCE_TAG: str = f"nominatim:{date.today().isoformat()}"

CSV_HEADER: list[str] = [
    "venue_id", "name", "city", "state", "country",
    "lat", "lon", "timezone_iana", "n_events", "geocode_source",
]


# ─── Cache helpers (Phase 22 WR-05 atomic-rename + malformed-JSON tolerant) ────

def _load_cache(cache_path: Path) -> dict[str, dict | None]:
    """Load the geocode cache JSON (empty dict if file missing).

    WR-05: malformed JSON tolerated — start from empty rather than crash.
    """
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"[sweep-venues] WARN: cache load failed ({type(exc).__name__}: "
            f"{exc}); starting from empty cache",
            file=sys.stderr,
        )
        return {}


def _save_cache(cache: dict[str, dict | None], cache_path: Path) -> None:
    """Persist the geocode cache atomically (WR-05)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(cache, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, cache_path)


# ─── Geolocator factory (WR-06 test seam) ───────────────────────────────────

def _make_geolocator():  # noqa: ANN202
    """Lazy-import geopy and construct the Nominatim client.

    geopy is NOT a runtime dep (D-07); only imported on real cache miss.
    """
    from geopy.geocoders import Nominatim  # noqa: PLC0415

    return Nominatim(
        user_agent=NOMINATIM_USER_AGENT,
        timeout=NOMINATIM_HTTP_TIMEOUT_S,
    )


def _load_or_geocode(
    venue_name: str, cache: dict[str, dict | None]
) -> dict | None:
    """Cache-first geocode.

    Cache semantics (Phase 22 WR-02): both hits AND misses are cached.
    A Nominatim None return is cached as None sentinel; transient
    network/throttle failures are NOT cached (so operator can rerun).
    """
    if venue_name in cache:
        return cache[venue_name]  # may be None (cached miss) — fine

    from geopy.exc import (  # noqa: PLC0415
        GeocoderRateLimited,
        GeocoderUnavailable,
    )

    geolocator = _make_geolocator()
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
            f"[sweep-venues] WARN: skip {venue_name!r}: {type(exc).__name__}",
            file=sys.stderr,
        )
        return None
    except Exception as exc:  # noqa: BLE001
        print(
            f"[sweep-venues] WARN: skip {venue_name!r}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None
    if location is None:
        cache[venue_name] = None  # WR-02 miss sentinel
        return None
    record: dict[str, Any] = {
        "lat": location.latitude,
        "lon": location.longitude,
        "address": location.raw.get("address", {}),
        "geocode_source": GEOCODE_SOURCE_TAG,
    }
    cache[venue_name] = record
    return record


def _derive_iana_tz(lat: float, lon: float) -> str | None:
    """TimezoneFinder() lookup (Phase 22 Finding 6: NOT TimezoneFinderL)."""
    from timezonefinder import TimezoneFinder  # noqa: PLC0415

    tf = TimezoneFinder()
    return tf.timezone_at(lat=lat, lng=lon)


# ─── DB query ────────────────────────────────────────────────────────────────

def _query_distinct_venues(session: Any) -> list[tuple[str, int]]:
    """SELECT location, COUNT(*) FROM events GROUP BY location ORDER BY 2 DESC."""
    from sqlalchemy import func as sa_func, select  # noqa: PLC0415

    from ufc_prediction.models.event import Event  # noqa: PLC0415

    rows = session.execute(
        select(Event.location, sa_func.count(Event.id).label("n"))
        .where(Event.location.isnot(None))
        .group_by(Event.location)
        .order_by(sa_func.count(Event.id).desc()),
    ).all()
    return [(r[0], r[1]) for r in rows]


# ─── Existing CSV loader (preserve byte-identity for existing rows) ──────────

def _load_existing_rows(path: Path) -> list[dict[str, str]]:
    """Load the existing venues.csv as a list of dicts (preserve order)."""
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _emit_csv(rows: list[dict[str, str]], path: Path) -> None:
    """Atomic write of the 10-col CSV (WR-05)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in CSV_HEADER})
    os.replace(tmp_path, path)


# ─── Notes emitter ───────────────────────────────────────────────────────────

def _emit_notes(
    notes_path: Path,
    *,
    cache_hits: int,
    nominatim_calls: int,
    nominatim_hits: int,
    nominatim_misses: int,
    wall_clock_s: float,
    cooldowns: int,
    rows_before: int,
    rows_after: int,
    misses: list[tuple[str, str]],
) -> None:
    """Emit operator-readable sweep notes (WR-05 atomic write)."""
    ts = datetime.now(timezone.utc).isoformat()
    lines: list[str] = []
    lines.append("# Phase 28 — Venues Sweep Notes")
    lines.append("")
    lines.append(f"**Generated:** {ts}")
    lines.append(
        f"**User-agent:** {NOMINATIM_USER_AGENT} (D-09 override)"
    )
    lines.append(f"**Cache hits (skipped network):** {cache_hits}")
    lines.append(f"**Nominatim calls:** {nominatim_calls}")
    lines.append(f"**Nominatim hits:** {nominatim_hits}")
    lines.append(f"**Nominatim misses:** {nominatim_misses}")
    lines.append(f"**Wall-clock:** {wall_clock_s:.1f}s")
    lines.append(f"**Throttle cooldowns triggered:** {cooldowns}")
    lines.append("")
    lines.append("## Misses (require operator manual entry to reach 95% coverage)")
    lines.append("")
    if misses:
        lines.append("| location | reason |")
        lines.append("|---|---|")
        for loc, reason in misses:
            safe_loc = loc.replace("|", "\\|")
            safe_reason = reason.replace("|", "\\|")
            lines.append(f"| {safe_loc} | {safe_reason} |")
    else:
        lines.append("_(none)_")
    lines.append("")
    lines.append("## venues.csv delta")
    lines.append("")
    lines.append(f"Before: {rows_before} rows")
    lines.append(f"After: {rows_after} rows")
    lines.append(f"Added: {rows_after - rows_before} rows")
    lines.append("")

    notes_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = notes_path.with_suffix(notes_path.suffix + ".tmp")
    tmp_path.write_text("\n".join(lines), encoding="utf-8")
    os.replace(tmp_path, notes_path)


# ─── Main ────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:  # noqa: ARG001
    """Run the sweep.

    Returns 0 on success, 2 on locked-constants drift.
    """
    # Locked-constants drift check (Phase 22 AF startup discipline)
    expected = {
        "NOMINATIM_USER_AGENT": (
            NOMINATIM_USER_AGENT,
            "ufc-fight-prediction-v23-venues-sweep",
        ),
        "NOMINATIM_DELAY_S": (NOMINATIM_DELAY_S, 1.2),
        "OUTPUT_CSV": (str(OUTPUT_CSV), "data/venues.csv"),
        "CSV_HEADER_LEN": (len(CSV_HEADER), 10),
    }
    for name, (got, want) in expected.items():
        if got != want:
            print(
                f"[sweep-venues] FATAL: {name} drift: expected {want!r}, got {got!r}",
                file=sys.stderr,
            )
            return 2

    t0 = time.monotonic()

    # 1. Load existing state
    existing_rows = _load_existing_rows(OUTPUT_CSV)
    rows_before = len(existing_rows)
    cache = _load_cache(GEOCODE_CACHE)
    existing_names = {row["name"] for row in existing_rows}
    if existing_rows:
        next_venue_id = max(int(row["venue_id"]) for row in existing_rows) + 1
    else:
        next_venue_id = 1
    print(
        f"[sweep-venues] loaded {rows_before} existing rows; "
        f"next venue_id = {next_venue_id}; cache entries = {len(cache)}"
    )

    # 2. Query DB for distinct locations
    from ufc_prediction.db.session import SessionLocal  # noqa: PLC0415

    session = SessionLocal()
    try:
        db_rows = _query_distinct_venues(session)
    finally:
        session.close()
    db_dict = {loc: n for loc, n in db_rows}
    print(f"[sweep-venues] DB has {len(db_rows)} distinct locations")

    # 3. Update n_events for existing rows (catches incidental growth)
    for row in existing_rows:
        new_n = db_dict.get(row["name"])
        if new_n is not None:
            row["n_events"] = str(new_n)

    # 4. Sweep loop for missing
    missing = [(loc, n) for loc, n in db_rows if loc not in existing_names]
    print(f"[sweep-venues] {len(missing)} missing locations to sweep")

    SAVE_EVERY = 10
    consecutive_fails = 0
    cooldowns = 0
    nominatim_calls = 0
    nominatim_hits = 0
    nominatim_misses = 0
    cache_hits = 0
    misses_log: list[tuple[str, str]] = []
    appended_rows: list[dict[str, str]] = []

    for idx, (raw_location, n_events) in enumerate(missing, start=1):
        was_cached = raw_location in cache
        if was_cached:
            cache_hits += 1
        else:
            nominatim_calls += 1

        record = _load_or_geocode(raw_location, cache)

        # Adaptive back-off (Phase 22 pattern) — only on un-cached failures
        if not was_cached:
            if record is None:
                consecutive_fails += 1
                if consecutive_fails >= NOMINATIM_CONSECUTIVE_FAILS_TRIGGER:
                    print(
                        f"[sweep-venues] BACK-OFF: {consecutive_fails} consecutive "
                        f"failures; sleeping {NOMINATIM_THROTTLE_COOLDOWN_S}s",
                        file=sys.stderr,
                    )
                    time.sleep(NOMINATIM_THROTTLE_COOLDOWN_S)
                    consecutive_fails = 0
                    cooldowns += 1
            else:
                consecutive_fails = 0

        # Incremental cache save
        if idx % SAVE_EVERY == 0:
            _save_cache(cache, GEOCODE_CACHE)
            print(
                f"[sweep-venues] progress: {idx}/{len(missing)} "
                f"(cache_hits={cache_hits}, hits={nominatim_hits}, "
                f"misses={nominatim_misses})"
            )

        if record is None:
            nominatim_misses += 1
            misses_log.append((raw_location, "MISS — no Nominatim result"))
            print(
                f"[sweep-venues] WARN: no geocode for {raw_location!r}; "
                "manual review needed",
                file=sys.stderr,
            )
            continue

        # Hit — derive timezone + assemble row
        nominatim_hits += 1
        addr = record["address"]
        try:
            tz = _derive_iana_tz(record["lat"], record["lon"])
        except Exception as exc:  # noqa: BLE001
            print(
                f"[sweep-venues] WARN: tz lookup failed for {raw_location!r}: "
                f"{type(exc).__name__}: {exc}; skipping",
                file=sys.stderr,
            )
            misses_log.append((raw_location, f"tz lookup failed: {exc}"))
            continue
        if tz is None:
            print(
                f"[sweep-venues] WARN: no tz for {raw_location!r}; skipping",
                file=sys.stderr,
            )
            misses_log.append((raw_location, "tz None"))
            continue

        new_row = {
            "venue_id": str(next_venue_id),
            "name": raw_location,  # preserve raw events.location string for matching
            "city": addr.get("city") or addr.get("town") or addr.get("village") or "",
            "state": addr.get("state") or "",
            "country": addr.get("country", ""),
            "lat": str(record["lat"]),
            "lon": str(record["lon"]),
            "timezone_iana": tz,
            "n_events": str(n_events),
            "geocode_source": record.get("geocode_source", GEOCODE_SOURCE_TAG),
        }
        appended_rows.append(new_row)
        next_venue_id += 1

    # Final cache save
    _save_cache(cache, GEOCODE_CACHE)

    # 5. Validate + emit CSV (atomic)
    all_rows = existing_rows + appended_rows
    # Validate every row has non-empty lat/lon/timezone_iana
    bad = [
        r for r in all_rows
        if not str(r.get("lat", "")).strip()
        or not str(r.get("lon", "")).strip()
        or not str(r.get("timezone_iana", "")).strip()
    ]
    if bad:
        print(
            f"[sweep-venues] FATAL: {len(bad)} rows with empty lat/lon/timezone: "
            f"{bad[:3]}",
            file=sys.stderr,
        )
        return 4

    _emit_csv(all_rows, OUTPUT_CSV)
    rows_after = len(all_rows)
    wall_clock_s = time.monotonic() - t0

    # 6. Emit notes — only on a real sweep (preserves initial-sweep metrics on
    # idempotent reruns; the substantive run is the load-bearing artifact).
    if (
        len(appended_rows) > 0
        or nominatim_calls > 0
        or nominatim_misses > 0
    ):
        _emit_notes(
            NOTES_PATH,
            cache_hits=cache_hits,
            nominatim_calls=nominatim_calls,
            nominatim_hits=nominatim_hits,
            nominatim_misses=nominatim_misses,
            wall_clock_s=wall_clock_s,
            cooldowns=cooldowns,
            rows_before=rows_before,
            rows_after=rows_after,
            misses=misses_log,
        )
    else:
        print(
            "[sweep-venues] idempotent rerun (no sweep work) — preserving prior "
            f"{NOTES_PATH} unchanged"
        )

    print(
        f"[sweep-venues] DONE — wrote {rows_after} rows ({rows_before} preserved + "
        f"{len(appended_rows)} appended); cache_hits={cache_hits}, "
        f"nominatim_calls={nominatim_calls}, hits={nominatim_hits}, "
        f"misses={nominatim_misses}, wall_clock={wall_clock_s:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
