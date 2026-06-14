#!/usr/bin/env python
"""INGEST-V23-01 slim driver — full-corpus referee backfill (Phase 28-02).

Mirrors ``scripts/audit_referees.py`` (Phase 22) but writes through
``upsert_referee`` + CR-01-guarded ``events.referee_id`` set instead of
counting + emitting an audit JSON. The audit driver iterates a deterministic
200-event sample; this driver iterates EVERY event with ``referee_id IS NULL``.

**Pitfall #2 (Phase 28 research)** — we DO NOT call ``scrape_all_events()``.
That re-runs the full ingest pipeline (fighter profiles, round_stats,
fight upserts, etc.) which is wildly overkill for a referee-only backfill
and would burn ~hours of UFCStats quota for one column. This slim driver
only fetches event-detail + fight-detail HTML to extract the referee, then
applies the CR-01 guard at ``ingest.py:470-474``.

**CR-01 binding (Phase 22)** — never overwrite a previously-set
``events.referee_id``. The aggregation block in ``_resolve_event_referee``
is a line-for-line mirror of ``src/ufc_prediction/scraper/ingest.py:470-474``.

**External-blocker policy** — if UFCStats throttles (≥3 consecutive
429/timeout failures), the loop logs WARN, leaves the event's ``referee_id``
NULL, and continues. Surfacing via coverage report is Plan 28-03 work.

**Banned imports per Pitfall #1 / Finding 11** — nothing under
``ufc_prediction.ml.*`` (LIVE-03 + AUDIT-01 byte-identity guard).

Usage:
    uv run python scripts/scrape_referees_full.py --dry-run
    uv run python scripts/scrape_referees_full.py --confirm
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


# ── Locked constants (Phase 28 CONTEXT D-04 + Phase 22 audit_referees.py) ──

HTTP_DELAY_SECONDS_DEFAULT: float = 1.2
HTTP_TIMEOUT_SECONDS: float = 10.0
HTTP_MAX_RETRIES: int = 2
WORKERS_DEFAULT: int = 4

CACHE_DIR: Path = Path("data/ufcstats_event_detail_cache")
FIGHT_CACHE_DIR: Path = CACHE_DIR / "fights"

_CACHE_KEY_RE = re.compile(r"[A-Za-z0-9_-]+")

logger = logging.getLogger(__name__)


# ── Cache-first HTTP fetch (lifted from scripts/audit_referees.py:112-157) ─


def _read_cached_html(cache_dir: Path, cache_key: str) -> str | None:
    """Return cached HTML for ``cache_key`` if present and key is safe; else None.

    Path-traversal guard: ``re.fullmatch(r"[A-Za-z0-9_-]+", cache_key)``.
    """
    if not _CACHE_KEY_RE.fullmatch(cache_key):
        logger.warning("[scrape-referees] cache_key rejected (key=%r); skipping cache read.", cache_key)
        return None
    cache_path = cache_dir / f"{cache_key}.html"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    return None


def _write_cached_html(
    cache_dir: Path,
    cache_key: str,
    html: str,
    cache_write_failures: list[str] | None = None,
) -> None:
    """Best-effort cache write keyed by cache_key. Mirrors path-safety of _read_cached_html."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not _CACHE_KEY_RE.fullmatch(cache_key):
        return  # key already rejected upstream
    cache_path = cache_dir / f"{cache_key}.html"
    if not html:
        return
    try:
        cache_path.write_text(html, encoding="utf-8")
    except OSError:
        logger.warning("[scrape-referees] cache write failed for %s", cache_key)
        if cache_write_failures is not None:
            cache_write_failures.append(cache_key)


def _load_or_fetch_html(
    client,
    *,
    cache_key: str,
    source_url: str,
    cache_dir: Path,
    cache_write_failures: list[str] | None = None,
) -> str | None:
    """Cache-first single-fetch keyed by cache_key. Re-runs only re-fetch failures.

    Path-traversal guard: ``re.fullmatch(r"[A-Za-z0-9_-]+", cache_key)`` —
    REQUIRED before any disk-path interpolation. Mirrors the discipline of
    ``audit_referees._load_or_fetch_html`` (T-22-02-XX threat model in
    Phase 22 + Phase 28 T-28-02-01).

    Kept for the per-event ``event_html`` path (single fetch); per-fight
    fetches now go through the worker pool via ``client.map_get()`` —
    see ``_run_confirmed_scrape`` for the parallel path.
    """
    if not _CACHE_KEY_RE.fullmatch(cache_key):
        logger.warning("[scrape-referees] cache_key rejected (key=%r); skipping URL.", cache_key)
        return None
    cached = _read_cached_html(cache_dir, cache_key)
    if cached is not None:
        return cached
    try:
        html = client.get(source_url)
    except Exception as exc:  # noqa: BLE001 — fetch failures are expected; log and bail
        logger.warning("[scrape-referees] fetch failed for %s: %s", source_url, exc)
        return None
    if html:
        _write_cached_html(cache_dir, cache_key, html, cache_write_failures)
    return html


# ── DB queries + CR-01 aggregation block ──────────────────────────────────


def _events_needing_referee(session) -> Iterable:
    """All events WHERE referee_id IS NULL AND source_url IS NOT NULL.

    Returns an iterable of Event ORM objects in id-order (deterministic for
    idempotent re-runs). Empty iterator if substrate is already complete.
    """
    from sqlalchemy import select  # noqa: PLC0415 — lazy import; keeps unit cost low

    from ufc_prediction.models.event import Event  # noqa: PLC0415

    return session.execute(
        select(Event)
        .where(Event.referee_id.is_(None))
        .where(Event.source_url.isnot(None))
        .order_by(Event.id)
    ).scalars().all()


def _resolve_event_referee(
    session,
    db_event,
    raw_names: list[str],
) -> tuple[int | None, str | None]:
    """CR-01-guarded referee resolution + assignment.

    Mirrors ``src/ufc_prediction/scraper/ingest.py:470-474`` line-for-line.
    Returns ``(new_ref_id, raw_name)`` for telemetry. Sets
    ``db_event.referee_id`` ONLY if currently NULL.
    """
    from ufc_prediction.data.upsert import upsert_referee  # noqa: PLC0415

    if not raw_names:
        return (None, None)
    most_frequent = Counter(raw_names).most_common(1)[0][0]
    new_ref_id = upsert_referee(session, raw_name=most_frequent)
    if db_event.referee_id is None and new_ref_id is not None:
        db_event.referee_id = new_ref_id  # CR-01: set-once
    return (new_ref_id, most_frequent)


def _count_cache_hits_expected(events, *, cache_dir: Path) -> int:
    """Count how many events already have their event-detail HTML cached.

    Drives the dry-run ETA: cache hits are instantaneous (~10ms each); cache
    misses pay the 1.2s rate-limit. We approximate by counting only
    event-detail cache hits — fight-detail caches are a separate (and larger)
    consideration tracked in the audit_referees logs.
    """
    if not cache_dir.exists():
        return 0
    count = 0
    for ev in events:
        cache_key = str(ev.id)
        if not _CACHE_KEY_RE.fullmatch(cache_key):
            continue
        if (cache_dir / f"{cache_key}.html").exists():
            count += 1
    return count


def _print_dry_run_summary(
    events_to_process: int,
    cache_hits_expected: int,
    delay: float,
    workers: int,
) -> None:
    """Print the dry-run preview to stdout (CONTEXT D-08 operator-friendly)."""
    cache_misses = max(0, events_to_process - cache_hits_expected)
    # Approximate: cache hits are ~0s; cache misses pay delay/workers per
    # request. Aggregate throughput = workers/delay req/s.
    #
    # FIX (scrape-referees-slow debug): per event, we also fetch ~10 fight-
    # detail pages (each event has ~10-13 fights). The original ETA assumed
    # ONE request per event and ignored the fight-detail fan-out, undershooting
    # by ~10×. Compound that with the original serial path (workers=4 declared
    # but unused) and the measured rate was ~50× slower than predicted. With
    # the worker pool now wired via client.map_get() the fight fetches are
    # actually parallel, so this corrected estimate is reachable.
    avg_fights_per_event = 10
    if events_to_process == 0:
        eta_minutes_float = 0.0
    else:
        throughput = max(1, workers) / max(0.01, delay)  # req/s
        event_fetches = cache_misses                      # event-detail pages
        fight_fetches = events_to_process * avg_fights_per_event  # all fights, conservative
        # Subtract anything already cached (fight-detail cache hits are ~free).
        fight_misses_estimate = fight_fetches  # upper bound; the cache helps in practice
        total_fetches = event_fetches + fight_misses_estimate
        eta_seconds = total_fetches / throughput
        eta_minutes_float = eta_seconds / 60.0
    print("[scrape-referees] DRY-RUN preview (no network calls, no DB writes)")
    print(f"[scrape-referees] Events to process: {events_to_process}")
    print(f"[scrape-referees] Cache hits expected: {cache_hits_expected}")
    print(f"[scrape-referees] Cache misses (network fetches): {cache_misses}")
    print(f"[scrape-referees] Rate limit: {delay}s/request, workers={workers}")
    print(f"[scrape-referees] ETA: {eta_minutes_float:.1f} minutes")
    print("[scrape-referees] Re-invoke with --confirm to actually scrape.")


# ── main() ────────────────────────────────────────────────────────────────


def main(
    *,
    delay: float = HTTP_DELAY_SECONDS_DEFAULT,
    workers: int = WORKERS_DEFAULT,
    dry_run: bool = False,
    confirm: bool = False,
    output_coverage_path: Path | None = None,
) -> int:
    """Returns exit code (0 success, 1 user-error, 2 fatal).

    Iterates events WHERE ``referee_id IS NULL``, cache-first fetches event +
    fight HTML, aggregates referees via Counter.most_common, applies
    CR-01-guarded upsert. Per-event commit (Phase 22 convention; mid-run-crash
    recovery + idempotent re-runs).

    --dry-run: prints {events_to_process, cache_hits_expected, eta_minutes},
    returns 0. NO network calls; NO DB writes.

    --confirm absent AND not --dry-run: prints safety message, returns 1.

    output_coverage_path: reserved for Plan 28-03 integration. Currently
    unused (only the venue CLI emits an unmatched-venues artifact in 28-02).
    """
    _ = output_coverage_path  # reserved for Plan 28-03 — silence linter
    logging.basicConfig(level=logging.INFO, format="[scrape-referees] %(message)s")

    # Pre-flight: open session, count NULL-referee events.
    from ufc_prediction.db.session import SessionLocal  # noqa: PLC0415

    session = SessionLocal()
    try:
        events = list(_events_needing_referee(session))
        events_to_process = len(events)
        cache_hits_expected = _count_cache_hits_expected(events, cache_dir=CACHE_DIR)

        # Pre-flight reachability check.
        if events_to_process == 0:
            print(
                "[scrape-referees] Substrate already complete — 0 events with "
                "NULL referee_id. Exiting 0."
            )
            return 0

        # Dry-run path — print and exit.
        if dry_run:
            _print_dry_run_summary(events_to_process, cache_hits_expected, delay, workers)
            return 0

        # Safety gate (CONTEXT D-08).
        if not confirm:
            print(
                "[scrape-referees] SAFETY GATE — re-invoke with --confirm to "
                "actually scrape, or --dry-run for a preview (CONTEXT D-08). "
                "Refusing to proceed without explicit confirmation."
            )
            return 1

        # Confirmed path — real scrape begins here.
        return _run_confirmed_scrape(
            session,
            events=events,
            delay=delay,
            workers=workers,
        )
    finally:
        session.close()


def _run_confirmed_scrape(
    session,
    *,
    events: list,
    delay: float,
    workers: int,
) -> int:
    """Confirmed-path scrape loop. Returns exit code."""
    from ufc_prediction.scraper.client import ScraperClient  # noqa: PLC0415
    from ufc_prediction.scraper.parse_event_detail import parse_event_detail  # noqa: PLC0415
    from ufc_prediction.scraper.parse_fight_detail import parse_fight_detail  # noqa: PLC0415

    spike_started = datetime.now(timezone.utc)
    client = ScraperClient(
        delay=delay,
        max_retries=HTTP_MAX_RETRIES,
        timeout=HTTP_TIMEOUT_SECONDS,
        workers=workers,
    )
    cache_write_failures: list[str] = []
    resolved = 0
    unresolved = 0
    consecutive_failures = 0
    abort_threshold = 3

    print(f"[scrape-referees] Begin scrape ({len(events)} events @ {delay}s rate-limit, workers={workers})...")
    for i, ev in enumerate(events, start=1):
        if i % 10 == 0:
            print(
                f"[scrape-referees] Progress {i}/{len(events)} "
                f"(resolved={resolved}, unresolved={unresolved})..."
            )
        event_html = _load_or_fetch_html(
            client,
            cache_key=str(ev.id),
            source_url=ev.source_url,
            cache_dir=CACHE_DIR,
            cache_write_failures=cache_write_failures,
        )
        if event_html is None:
            consecutive_failures += 1
            unresolved += 1
            if consecutive_failures >= abort_threshold:
                print(
                    f"[scrape-referees] WARN: {consecutive_failures} consecutive "
                    "fetch failures — UFCStats may be throttling. Continuing with "
                    "remaining events; failed events stay NULL."
                )
            continue
        consecutive_failures = 0
        try:
            event_detail = parse_event_detail(event_html, ev.source_url)
        except Exception as exc:  # noqa: BLE001 — slim driver tolerates parser drift
            logger.warning("[scrape-referees] parse_event_detail failed for event_id=%s: %s", ev.id, exc)
            unresolved += 1
            continue
        # Per-fight referee extraction.
        # FIX (scrape-referees-slow debug): originally fetched fight HTML
        # serially inside this loop — `ScraperClient(workers=4)` was created
        # but its worker pool only fires inside `.map()`/`.map_get()`. A
        # sequential `client.get()` chain serializes on the calling thread,
        # so ~10 fights × 1.2s/fight × 770 events ≈ 154 min, not the
        # dry-run's claimed 2.9 min. Split into two passes: (1) gather
        # cached HTML + missing URLs serially (free); (2) batch-fetch
        # missing URLs through the worker pool. Per-event commit cadence
        # preserved at the bottom of the loop (CR-01 idempotency intact).
        fight_jobs: list[tuple[str, str]] = []  # (cache_key, source_url)
        for fight in event_detail.fights:
            fight_url = getattr(fight, "fight_url", "") or ""
            if not fight_url:
                continue
            fight_slug = fight_url.rstrip("/").rsplit("/", 1)[-1]
            if not fight_slug:
                continue
            fight_jobs.append((fight_slug, fight_url))

        # Pass 1: serve from cache where possible (free; no HTTP).
        fight_htmls: list[str | None] = [None] * len(fight_jobs)
        missing_idx: list[int] = []
        missing_urls: list[str] = []
        for j, (fight_slug, fight_url) in enumerate(fight_jobs):
            cached = _read_cached_html(FIGHT_CACHE_DIR, fight_slug)
            if cached is not None:
                fight_htmls[j] = cached
            else:
                missing_idx.append(j)
                missing_urls.append(fight_url)

        # Pass 2: parallel-fetch the cache misses through the worker pool.
        if missing_urls:
            try:
                fetched = client.map_get(missing_urls)
            except Exception as exc:  # noqa: BLE001 — slim driver tolerates HTTP layer faults
                logger.warning(
                    "[scrape-referees] map_get failed for event_id=%s (%d urls): %s",
                    ev.id, len(missing_urls), exc,
                )
                fetched = [None] * len(missing_urls)
            for j_local, html in zip(missing_idx, fetched, strict=True):
                fight_htmls[j_local] = html
                if html is not None:
                    fight_slug = fight_jobs[j_local][0]
                    _write_cached_html(
                        FIGHT_CACHE_DIR, fight_slug, html, cache_write_failures,
                    )

        # Pass 3: parse + collect referee names.
        raw_names: list[str] = []
        cardinality_failures_this_event = 0
        for j, ((fight_slug, _fight_url), fight_html) in enumerate(zip(fight_jobs, fight_htmls, strict=True)):
            if fight_html is None:
                continue
            try:
                fight_detail = parse_fight_detail(fight_html)
            except Exception as exc:  # noqa: BLE001 — slim driver tolerates parser drift
                msg = str(exc)
                logger.warning(
                    "[scrape-referees] parse_fight_detail failed event_id=%s slug=%s: %s",
                    ev.id, fight_slug, msg,
                )
                # Short-circuit upcoming/future events: their fight-detail
                # pages return "expected >=1 data sections, got 0". After
                # 2 such failures in one event, treat the remaining fights
                # as future-state and skip them — no signal to extract.
                if "data sections, got 0" in msg:
                    cardinality_failures_this_event += 1
                    if cardinality_failures_this_event >= 2:
                        logger.info(
                            "[scrape-referees] event_id=%s appears to be a "
                            "future/incomplete event; skipping remaining %d fights",
                            ev.id, len(fight_jobs) - j - 1,
                        )
                        break
                continue
            if fight_detail.referee:
                raw_names.append(fight_detail.referee)

        new_ref_id, _raw = _resolve_event_referee(session, ev, raw_names)
        # Per-event commit (Phase 22 convention) — mid-run-crash recovery.
        session.commit()
        if new_ref_id is not None and ev.referee_id is not None:
            resolved += 1
        else:
            unresolved += 1

    spike_finished = datetime.now(timezone.utc)
    duration_s = (spike_finished - spike_started).total_seconds()
    print(
        f"[scrape-referees] Scrape complete. resolved={resolved}, "
        f"unresolved={unresolved}, total={len(events)}, "
        f"duration={duration_s:.0f}s, cache_write_failures={len(cache_write_failures)}"
    )
    return 0


def _parse_argv(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="INGEST-V23-01 slim driver — full-corpus referee backfill (Phase 28-02)."
    )
    parser.add_argument("--delay", type=float, default=HTTP_DELAY_SECONDS_DEFAULT)
    parser.add_argument("--workers", type=int, default=WORKERS_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_argv()
    sys.exit(
        main(
            delay=args.delay,
            workers=args.workers,
            dry_run=args.dry_run,
            confirm=args.confirm,
        )
    )
