#!/usr/bin/env python
"""REF-V22-00 referee coverage audit (Phase 22).

Per CONTEXT.md REVISION-02 + RESEARCH Finding 2: referee field is parsed at
parse_fight_detail.py:82 into Pydantic FightDetailPage.referee but NEVER persisted
to the DB (verified by grep: zero referee references in data/upsert.py / models/*.py).
SQL groupby is therefore impossible; this driver implements PATH 3 (sample-and-fetch
200 random events at 1.2s rate-limit, ~4 min minimum wall-clock per event-fetch;
total may be 8-12 min once per-fight fetches are added).

Outputs REF_00_AUDIT.json with mechanical scope_recommendation per CONTEXT D-01:
  top30_coverage_rate >= 0.60 -> "proceed"   (Plan 22-02 commits REF migration)
  top30_coverage_rate <  0.60 -> "drop"      (REF-V22-01 deferred to v2.3+ Backlog)

Banned imports (Pitfall #1 / Finding 11): nothing under ``ufc_prediction.ml.*``.
Wave-0 lint guard runs `! grep -rn 'from ufc_prediction\\.ml' scripts/audit_referees.py`.

Usage:
    uv run python scripts/audit_referees.py \\
        --output-path .planning/phases/22-ref-travel-camp-schema-scrapers-and-migrations/REF_00_AUDIT.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ufc_prediction.scraper.referee_normalize import normalize_referee_name


# ── Locked constants (CONTEXT D-01..D-02 + REVISION-02) ──────────────────

SAMPLE_SIZE: int = 200  # CONTEXT REVISION-02 path 3
RANDOM_STATE: int = 42  # v2.x convention (Phase 20 precedent)
REF_TOP30_COVERAGE_THRESHOLD: float = 0.60  # CONTEXT D-01 binary gate
HTTP_DELAY_SECONDS: float = 1.2  # UFCStats polite-delay (Phase 20 precedent)
HTTP_TIMEOUT_SECONDS: float = 10.0
HTTP_MAX_RETRIES: int = 2

CACHE_DIR: Path = Path("data/ufcstats_event_detail_cache")
OUTPUT_PATH_DEFAULT: Path = Path(
    ".planning/phases/22-ref-travel-camp-schema-scrapers-and-migrations/REF_00_AUDIT.json"
)
AUDIT_VERSION: str = "v22.00"
HIGH_FAILURE_RATE_THRESHOLD: float = 0.05  # >5% fetch-failures -> risk_flag


logger = logging.getLogger(__name__)


# ── Public helpers (exported for tests; underscore-prefixed but stable) ──


def _derive_scope_recommendation(*, top30_rate: float) -> str:
    """Mechanical CONTEXT D-01 derivation. Binary; no operator interpretation.

    >= 0.60 -> "proceed"   (REF-V22-01 migration commits in Plan 22-02)
    <  0.60 -> "drop"      (defer to v2.3+ Backlog per Phase 20 CAMP precedent)
    """
    if top30_rate >= REF_TOP30_COVERAGE_THRESHOLD:
        return "proceed"
    return "drop"


def _compute_top30_coverage(referee_counter: Counter[str]) -> float:
    """Top-30 fight-count concentration: sum(top-30 fight counts) / sum(all fight counts).

    Returns 0.0 on empty input. Mirrors the CAMP audit semantics (Phase 20).
    """
    total = sum(referee_counter.values())
    if total == 0:
        return 0.0
    top30 = sum(count for _, count in referee_counter.most_common(30))
    return top30 / total


def _sample_event_urls(
    session, *, sample_size: int, random_state: int
) -> list[tuple[int, str, str, str]]:
    """SELECT id, name, date, source_url FROM events WHERE source='ufcstats' AND source_url IS NOT NULL.

    Returns deterministically-sampled list of (event_id, name, date_iso, source_url).
    Uses Python random.Random(random_state).sample for reproducibility (NOT SQL ORDER BY RANDOM()
    which is non-deterministic across Postgres versions).
    """
    from sqlalchemy import text  # noqa: PLC0415 — lazy import; keeps test cost low

    rows = session.execute(
        text(
            "SELECT id, name, date, source_url FROM events "
            "WHERE source = 'ufcstats' AND source_url IS NOT NULL "
            "ORDER BY id"
        )
    ).all()
    if not rows:
        return []
    rng = random.Random(random_state)
    sample = rng.sample(rows, k=min(sample_size, len(rows)))
    return [(int(r[0]), str(r[1] or ""), str(r[2] or ""), str(r[3] or "")) for r in sample]


def _load_or_fetch_html(
    client,
    *,
    cache_key: str,
    source_url: str,
    cache_dir: Path,
    cache_write_failures: list[str] | None = None,
) -> str | None:
    """Cache-first fetch keyed by cache_key. Re-runs only re-fetch failures.

    Mirrors CAMP audit pattern: write fetched HTML to data/ufcstats_event_detail_cache/<key>.html
    so subsequent runs (or recovery from interrupted batch) hit cache.

    cache_key is a free-form sanitized string (event id or fight slug); callers MUST pass an
    integer-derived or alpha-numeric-only value to avoid path traversal (Pitfall #5 mitigation).

    cache_write_failures (WR-03): if provided, cache_keys whose successful fetch
    could NOT be persisted to disk are appended here. The caller surfaces the
    count separately in REF_00_AUDIT.json so operators can distinguish network
    fetch failures (next rerun WILL re-fetch the URL) from cache-write failures
    (next rerun WILL ALSO re-fetch even though the original fetch succeeded —
    typically a disk-full / permissions issue worth raising).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Defensive cache-key validation: only allow [A-Za-z0-9_-]
    import re  # noqa: PLC0415

    if not re.fullmatch(r"[A-Za-z0-9_-]+", cache_key):
        logger.warning("[audit-referees] cache_key rejected (key=%r); skipping URL.", cache_key)
        return None
    cache_path = cache_dir / f"{cache_key}.html"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    try:
        html = client.get(source_url)
    except Exception as exc:  # noqa: BLE001 — fetch failures are expected; log and bail
        logger.warning("[audit-referees] fetch failed for %s: %s", source_url, exc)
        return None
    if html:
        try:
            cache_path.write_text(html, encoding="utf-8")
        except OSError:
            logger.warning("[audit-referees] cache write failed for %s", cache_key)
            if cache_write_failures is not None:
                cache_write_failures.append(cache_key)
    return html


def _write_audit_json(
    output_path: Path,
    *,
    sample_size: int,
    n_fetched: int,
    n_fetch_failures: int,
    top30_coverage_rate: float,
    scope_recommendation: str,
    per_referee_top30: list[dict],
    unparseable_examples: list[str],
    risk_flags: list[str],
    n_cache_write_failures: int = 0,
) -> None:
    """Emit REF_00_AUDIT.json with the payload (CONTEXT specifics).

    WR-03: ``n_cache_write_failures`` is reported separately from
    ``n_fetch_failures``. The former counts disk-write failures (the URL fetched
    successfully but the HTML couldn't be persisted to cache — typically
    disk-full / permission issues; next rerun WILL re-fetch). The latter counts
    true network/parser fetch failures.
    """
    payload: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "sample_size": sample_size,
        "sample_random_state": RANDOM_STATE,
        "n_fetched": n_fetched,
        "n_fetch_failures": n_fetch_failures,
        "n_cache_write_failures": n_cache_write_failures,
        "top30_coverage_rate": round(top30_coverage_rate, 4),
        "scope_recommendation": scope_recommendation,
        "per_referee_top30": per_referee_top30,
        "unparseable_examples": unparseable_examples,
        "risk_flags": risk_flags,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# ── argparse + main() ────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "REF-V22-00 referee coverage audit (Phase 22) — sample-and-fetch 200 "
            "random UFCStats events at 1.2s rate-limit; emit REF_00_AUDIT.json "
            "with mechanical scope_recommendation per CONTEXT D-01."
        ),
    )
    parser.add_argument(
        "--output-path",
        default=str(OUTPUT_PATH_DEFAULT),
        help="Output path for REF_00_AUDIT.json (default: per CONTEXT specifics).",
    )
    args = parser.parse_args(argv)

    output_path = Path(args.output_path)
    logging.basicConfig(level=logging.INFO, format="[audit-referees] %(message)s")

    spike_started = datetime.now(timezone.utc)

    # ── AF startup assertions (locked-constants drift detection) ──────────
    expected = {
        "SAMPLE_SIZE": (SAMPLE_SIZE, 200),
        "RANDOM_STATE": (RANDOM_STATE, 42),
        "REF_TOP30_COVERAGE_THRESHOLD": (REF_TOP30_COVERAGE_THRESHOLD, 0.60),
        "AUDIT_VERSION": (AUDIT_VERSION, "v22.00"),
    }
    for name, (got, want) in expected.items():
        if got != want:
            print(
                f"[audit-referees] FATAL: {name} drift: expected {want}, got {got}",
                file=sys.stderr,
            )
            return 2
    print(
        "[audit-referees] AF OK: SAMPLE_SIZE=200, RANDOM_STATE=42, "
        "REF_TOP30_COVERAGE_THRESHOLD=0.60, AUDIT_VERSION=v22.00."
    )

    # ── Lazy imports for DB + scraper (skip cost for AF-only invocation) ──
    from ufc_prediction.db.session import SessionLocal  # noqa: PLC0415
    from ufc_prediction.scraper.client import ScraperClient  # noqa: PLC0415
    from ufc_prediction.scraper.parse_event_detail import (  # noqa: PLC0415
        parse_event_detail,
    )
    from ufc_prediction.scraper.parse_fight_detail import (  # noqa: PLC0415
        parse_fight_detail,
    )

    print("[audit-referees] Loading event sample from DB...")
    t0 = time.time()
    session = SessionLocal()
    try:
        sample = _sample_event_urls(session, sample_size=SAMPLE_SIZE, random_state=RANDOM_STATE)
    finally:
        session.close()
    print(
        f"[audit-referees] Sampled {len(sample)} events in "
        f"{time.time() - t0:.1f}s (seed={RANDOM_STATE})."
    )
    if not sample:
        print(
            "[audit-referees] FATAL: no events with ufcstats source_url in DB. "
            "Cannot audit empty corpus.",
            file=sys.stderr,
        )
        return 2

    # ── Per-event fetch + parse loop ──────────────────────────────────────
    client = ScraperClient(
        delay=HTTP_DELAY_SECONDS,
        max_retries=HTTP_MAX_RETRIES,
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    referee_counter: Counter[str] = Counter()
    alias_variations: dict[str, set[str]] = defaultdict(set)
    unparseable_examples: list[str] = []
    n_fetched = 0
    n_fetch_failures = 0
    cache_write_failures: list[str] = []  # WR-03: surface in REF_00_AUDIT.json

    fight_cache_dir = CACHE_DIR / "fights"

    print(
        f"[audit-referees] Begin fetch loop ({len(sample)} events @ "
        f"{HTTP_DELAY_SECONDS}s rate-limit)..."
    )
    for i, (event_id, event_name, _date_iso, source_url) in enumerate(sample, start=1):
        if i % 10 == 0:
            print(
                f"[audit-referees] Progress {i}/{len(sample)} "
                f"(fetched={n_fetched}, failed={n_fetch_failures})..."
            )
        event_html = _load_or_fetch_html(
            client,
            cache_key=str(event_id),
            source_url=source_url,
            cache_dir=CACHE_DIR,
            cache_write_failures=cache_write_failures,
        )
        if event_html is None:
            n_fetch_failures += 1
            continue
        n_fetched += 1
        try:
            event_detail = parse_event_detail(event_html, source_url)
        except Exception as exc:  # noqa: BLE001 — audit driver tolerates parser drift
            if len(unparseable_examples) < 5:
                unparseable_examples.append(f"event_id={event_id} parse_event_detail failed: {exc}")
            continue
        # Per-fight referee extraction
        for fight in event_detail.fights:
            fight_url = getattr(fight, "fight_url", "") or ""
            if not fight_url:
                continue
            # Use slug from URL as cache key (UFCStats fight URLs end with hex slug)
            fight_slug = fight_url.rstrip("/").rsplit("/", 1)[-1]
            if not fight_slug:
                continue
            fight_html = _load_or_fetch_html(
                client,
                cache_key=fight_slug,
                source_url=fight_url,
                cache_dir=fight_cache_dir,
                cache_write_failures=cache_write_failures,
            )
            if fight_html is None:
                continue
            try:
                fight_detail = parse_fight_detail(fight_html)
            except Exception as exc:  # noqa: BLE001 — audit driver tolerates parser drift
                if len(unparseable_examples) < 5:
                    unparseable_examples.append(
                        f"event_id={event_id} fight_slug={fight_slug} "
                        f"parse_fight_detail failed: {exc}"
                    )
                continue
            raw_ref = fight_detail.referee
            norm_ref = normalize_referee_name(raw_ref)
            if norm_ref is None:
                if raw_ref and len(unparseable_examples) < 5:
                    unparseable_examples.append(
                        f"event_id={event_id} fight_slug={fight_slug} "
                        f"raw_referee={raw_ref!r} normalized to None"
                    )
                continue
            referee_counter[norm_ref] += 1
            if raw_ref and raw_ref != norm_ref:
                alias_variations[norm_ref].add(raw_ref)
            else:
                alias_variations[norm_ref].add(raw_ref or norm_ref)

    # ── Aggregate top-30 coverage + scope recommendation ──────────────────
    top30_rate = _compute_top30_coverage(referee_counter)
    scope_recommendation = _derive_scope_recommendation(top30_rate=top30_rate)

    per_referee_top30 = [
        {
            "name": sorted(alias_variations[norm])[0] if alias_variations[norm] else norm,
            "normalized_name": norm,
            "fight_count": count,
            "alias_variations": sorted(alias_variations[norm]),
        }
        for norm, count in referee_counter.most_common(30)
    ]

    # ── Risk flags ────────────────────────────────────────────────────────
    risk_flags: list[str] = []
    if n_fetch_failures > HIGH_FAILURE_RATE_THRESHOLD * SAMPLE_SIZE:
        risk_flags.append("high_fetch_failure_rate")
    if sum(referee_counter.values()) == 0:
        risk_flags.append("no_referee_data_extracted")
    # WR-03: any cache-write failure is unusual (disk-full / permissions);
    # flag it so the operator notices before the next run re-fetches.
    if cache_write_failures:
        risk_flags.append("cache_write_failures_present")

    # ── Emit artifact ─────────────────────────────────────────────────────
    _write_audit_json(
        output_path,
        sample_size=SAMPLE_SIZE,
        n_fetched=n_fetched,
        n_fetch_failures=n_fetch_failures,
        top30_coverage_rate=top30_rate,
        scope_recommendation=scope_recommendation,
        per_referee_top30=per_referee_top30,
        unparseable_examples=unparseable_examples,
        risk_flags=risk_flags,
        n_cache_write_failures=len(cache_write_failures),
    )
    spike_finished = datetime.now(timezone.utc)

    print(
        f"\n[audit-referees] Audit complete. Wrote:\n"
        f"  {output_path}\n"
        f"[audit-referees] n_fetched / n_fetch_failures = {n_fetched} / "
        f"{n_fetch_failures}\n"
        f"[audit-referees] n_cache_write_failures       = "
        f"{len(cache_write_failures)}\n"
        f"[audit-referees] total_referee_fights         = "
        f"{sum(referee_counter.values())}\n"
        f"[audit-referees] distinct_referees            = {len(referee_counter)}\n"
        f"[audit-referees] top30_coverage_rate          = {top30_rate:.4f}\n"
        f"[audit-referees] scope_recommendation         = {scope_recommendation}\n"
        f"[audit-referees] duration                     = "
        f"{(spike_finished - spike_started).total_seconds():.0f}s"
    )
    if risk_flags:
        print(f"[audit-referees] risk_flags = {risk_flags}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
