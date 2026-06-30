#!/usr/bin/env python
"""Phase 21 Plan 21-01 Task 4 — BFO coverage backfill driver (scaffolding).

Operator-driven entry point for the v2.2 BFO backfill. Plan 21-01 ships ONLY
the scaffolding: locked constants + AF startup asserts + DB gap-query +
URL enumeration via BFO ``/search`` + ``BFO_V22_GAP.json`` emit + ``--dry-run``
mode + ``--resume`` flag stub. The actual live scrape (via
``BFOScraper.scrape_event_urls`` from Task 3) + ingest (via existing
``BFOOddsIngester``) + coverage delta (``BFO_V22_COVERAGE.json``) + AUDIT-01
mid/end checkpoints are Plan 21-02.

Locked decisions (CONTEXT.md REVISION + 21-RESEARCH.md):
  - D-01: BFO ``/search?query={event.name}`` per gap event; first
    ``/events/<slug>-<id>`` match wins.
  - D-01a: Disambiguation guard — require slug to start with ``ufc-``.
    Bellator/Strikeforce/etc. collisions are logged as
    ``DISAMBIGUATION_FAIL``.
  - D-03: Hard skip via ``Event.date >= 2007-06-01`` filter at gap-query
    time (Phase 20 reachability finding; pre-2007 is structurally
    unavailable).
  - D-07: Coverage target = 0.96 (locked from Phase 20
    ``BFO_ARCHIVE_REACHABILITY.md``).
  - D-08: ``ScraperClient(delay=1.2, max_retries=3, timeout=5.0)`` —
    matches Phase 20 audit; single-threaded only (concurrency unmeasured).
  - REVISION (21-RESEARCH.md Option A): no Alembic migration; per-event
    CSV staging in ``data/bfo/v22-backfill/<batch_id>/`` is the audit
    trail. Existing ``BFOOddsIngester`` reads the CSV unmodified.
  - 21-PATTERNS.md anti-pattern #3: do NOT call the legacy bulk
    scrape-all method on BFOScraper (it re-fetches the entire fighter
    corpus and misses gap-driven URLs); use scrape_event_urls instead.

Anti-features (BANNED):
  - AF-1: hyperparameter retuning. Backfill, not retrain.
  - AF-2: feature engineering. No new feature columns produced.
  - AF-3: model file mutation. ``models/xgb_v2.joblib`` byte-identity is
    verified by ``scripts/audit_xgb_v2_sha.sh`` mid + end of Phase 21.
  - AF-4: any invocation of the legacy bulk-scrape orchestrator method
    (anti-pattern #3 explicit guard in test_bfo_backfill.py).

D-09(P15) carry-forward: this driver never persists models; AUDIT-01 chain
runs in Plan 21-02.

Usage:
    # Dry-run mode (gap query + URL resolution only; emits BFO_V22_GAP.json):
    uv run python scripts/bfo_backfill.py --dry-run

    # Resume mode (Plan 21-02 implements full logic; flag stub here):
    uv run python scripts/bfo_backfill.py --resume

Plan 21-01 Task 5 operator-action checkpoint:
    1. Operator runs ``--dry-run`` mode
    2. Inspects ``BFO_V22_GAP.json`` for url_acquired count + per-year
       breakdown
    3. Approves to unblock Plan 21-02 (live scrape + ingest)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
import urllib.parse
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

# ── Locked constants (D-01..D-11 REVISION; CONTEXT.md) ──────────────────

BFO_ARCHIVE_START_DATE: date = date(2007, 6, 1)
BFO_TIMEOUT_SECONDS: int = 5
BFO_DELAY_SECONDS: float = 1.2
BFO_MAX_RETRIES: int = 3
TARGET_RATE_THRESHOLD: float = 0.96  # Phase 20 BFO_ARCHIVE_REACHABILITY.md
SCRAPE_BATCH_DIR: str = "data/bfo/v22-backfill"
DISAMBIGUATION_REQUIRE_UFC_PREFIX: bool = True  # CONTEXT.md D-01a
GAP_JSON_PATH: str = ".planning/phases/21-bfo-coverage-backfill/BFO_V22_GAP.json"

# AUDIT-01 chain (D-09(P15) carry-forward; xgb_v2 byte-identity gate).
# --scrape mode emits the mid SHA file BEFORE returning success; if the SHA
# diverges from the canonical baseline, --scrape exits 1 (HARD gate).
XGB_V2_MODEL_PATH: str = "models/xgb_v2.joblib"
AUDIT_01_BASELINE_SHA_PATH: str = ".planning/AUDIT-01-BASELINE-SHA.txt"
AUDIT_01_SHA_MID_PATH: str = ".planning/phases/21-bfo-coverage-backfill/21-XGB-V2-SHA-MID.txt"

# Plan 21-02 Task 4: --emit-coverage output path.
COVERAGE_JSON_PATH: str = ".planning/phases/21-bfo-coverage-backfill/BFO_V22_COVERAGE.json"

# BFO endpoints (mirror scraper module).
BFO_BASE: str = "https://www.bestfightodds.com"
SEARCH_URL_TEMPLATE: str = f"{BFO_BASE}/search?query={{query}}"

# Pattern: extract `/events/<slug>-<numeric_id>` from BFO /search HTML.
#
# HYG-V26-02 (Phase 49): the slug+id grammar is now imported from
# `ufc_prediction.scraper.bfo_scraper.EVENT_SLUG_ID_PATTERN` so a single
# source of truth lives in the scraper module — preventing the WR-01
# / CORPUS-V25-02 hardening from silently drifting between the two
# extractors. The `href="..."` wrapper is local to this raw-HTML search
# path; the inner grammar is shared.
from ufc_prediction.scraper.bfo_scraper import EVENT_SLUG_ID_PATTERN

_EVENT_HREF_RE = re.compile(rf'href="(/events/({EVENT_SLUG_ID_PATTERN}))"')

logger = logging.getLogger(__name__)


# ── AF startup asserts (D-13 carry-forward; testable hook) ──────────────


def _run_af_startup_asserts() -> None:
    """AF-style assertions — fail fast on locked-constant drift.

    Mirrors the Phase 20 ``audit_camp_v22.main`` pattern (lines 432-461):
    every locked constant validated BEFORE any DB or HTTP work begins.
    Tampering with a constant trips an ``AssertionError`` with a
    CONTEXT.md pointer in the message.
    """
    assert BFO_ARCHIVE_START_DATE == date(2007, 6, 1), (
        "Locked from Phase 20 reachability finding; do not move without re-running BFO-V22-00 audit"
    )
    assert TARGET_RATE_THRESHOLD == 0.96, (
        "Locked from Phase 20 BFO_ARCHIVE_REACHABILITY.md; "
        "post-measurement renegotiation is forbidden (D-18 carry-forward)"
    )
    assert DISAMBIGUATION_REQUIRE_UFC_PREFIX is True, (
        "CONTEXT.md D-01a — Bellator/Strikeforce slug collision guard must remain enabled"
    )
    assert BFO_DELAY_SECONDS >= 1.0, "BFO_DELAY_SECONDS < 1.0 risks captcha at scale"
    assert BFO_MAX_RETRIES >= 1
    # HYG-V26-02: single-source-of-truth pin for the BFO event slug+id
    # grammar. If `bfo_scraper.EVENT_SLUG_ID_PATTERN` drifts, this halts at
    # startup BEFORE any HTTP traffic, surfacing the divergence next to the
    # other locked-constant guards. See Phase 49 CONTEXT.md.
    assert EVENT_SLUG_ID_PATTERN == r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?-\d+", (
        "EVENT_SLUG_ID_PATTERN drift detected — Phase 49 HYG-V26-02 lock. "
        "Update both `bfo_scraper.EVENT_SLUG_ID_PATTERN` and this assert in "
        "the same commit if BFO's slug grammar genuinely changes."
    )


# ── DB query helpers ────────────────────────────────────────────────────


def _query_gap_events(session: Any) -> list[Any]:
    """Return post-cutoff events lacking BFO odds coverage.

    SQL contract:
        SELECT events.id, events.name, events.date
        FROM events
        WHERE events.date >= BFO_ARCHIVE_START_DATE
          AND events.id NOT IN (
              SELECT DISTINCT f.event_id
              FROM fights f
              JOIN fight_odds fo ON fo.fight_id = f.id
              WHERE fo.closing_implied_prob IS NOT NULL
          )

    The cutoff filter (D-03 hard skip) lives in the WHERE clause —
    pre-2007-06-01 events are excluded from BOTH the numerator and the
    denominator (per CONTEXT.md D-03 rationale; including them deflates
    the coverage metric).

    Returns:
        List of Row objects with attributes ``id``, ``name``, ``date``.
        Empty list when no gap exists (everything is covered).
    """
    # Lazy import — keeps the test module importable even when the DB
    # session machinery isn't available in the test environment.
    from sqlalchemy import select

    from ufc_prediction.models.event import Event
    from ufc_prediction.models.fight import Fight
    from ufc_prediction.models.fight_odds import FightOdds

    covered_subq = (
        select(Fight.event_id)
        .join(FightOdds, FightOdds.fight_id == Fight.id)
        .where(FightOdds.closing_implied_prob.is_not(None))
        .distinct()
    )
    stmt = (
        select(Event.id, Event.name, Event.date)
        .where(Event.date >= BFO_ARCHIVE_START_DATE)
        .where(Event.id.notin_(covered_subq))
        .order_by(Event.date)
    )
    return list(session.execute(stmt).all())


# ── URL enumeration via BFO /search (D-01 + D-01a) ──────────────────────


def _enumerate_bfo_url(
    client: Any,
    event_name: str,
) -> tuple[str | None, str]:
    """Resolve ``event_name`` → BFO event URL via ``/search``.

    Workflow:
      1. Build ``/search?query={urllib.parse.quote(event_name)}``
      2. Fetch via ``client.get`` (RuntimeError caught → EVENT_NOT_FOUND)
      3. Extract first ``/events/<slug>-<numeric_id>`` href via regex
      4. Apply D-01a UFC-prefix guard:
         - DISAMBIGUATION_REQUIRE_UFC_PREFIX is True AND slug doesn't
           start with "ufc-" → return ``(None, "DISAMBIGUATION_FAIL")``
      5. Otherwise return ``(absolute_url, "OK")``

    Returns:
        ``(url_or_None, status)`` where status ∈
        {``"OK"``, ``"EVENT_NOT_FOUND"``, ``"DISAMBIGUATION_FAIL"``}.
    """
    search_url = SEARCH_URL_TEMPLATE.format(query=urllib.parse.quote(event_name))
    try:
        html = client.get(search_url)
    except (RuntimeError, ValueError) as exc:
        logger.warning(
            "[bfo-backfill] /search fetch failed for %r: %s",
            event_name,
            exc,
        )
        return (None, "EVENT_NOT_FOUND")

    if html is None:
        return (None, "EVENT_NOT_FOUND")

    match = _EVENT_HREF_RE.search(html)
    if match is None:
        return (None, "EVENT_NOT_FOUND")

    href = match.group(1)
    slug = match.group(2)

    # D-01a UFC-prefix guard.
    if DISAMBIGUATION_REQUIRE_UFC_PREFIX and not slug.startswith("ufc-"):
        logger.info(
            "[bfo-backfill] disambiguation_fail for %r: matched slug %r is not ufc-prefixed",
            event_name,
            slug,
        )
        return (None, "DISAMBIGUATION_FAIL")

    return (f"{BFO_BASE}{href}", "OK")


# ── BFO_V22_GAP.json emit (Plan 21-01 Task 4 + Task 5 review artifact) ──


def _emit_gap_json(
    *,
    gap_events: list[Any],
    url_results: list[tuple[Any, str | None, str]],
    path: Path,
    batch_id: str,
) -> None:
    """Write the operator-review artifact to ``path``.

    Schema (per Plan 21-01 Task 4 spec):
        {
          "batch_id": "<YYYYMMDD-HHMMSS>",
          "gap_query": {
            "cutoff_date": "2007-06-01",
            "n_in_scope_events": N,
            "n_events_with_no_odds": N_missing,
            "n_events_with_partial_odds": N_partial
          },
          "url_resolution": {
            "url_acquired": N_ok,
            "event_not_found": N_not_found,
            "disambiguation_fail": N_disamb
          },
          "estimated_scrape_wall_clock_min": <int>,
          "per_year_breakdown": {
            "2007": {"n_events": N, "url_acquired": N, ...},
            ...
          },
          "next_step": "<operator-readable note>"
        }

    Args:
        gap_events: Output of ``_query_gap_events`` (Row-likes with
            ``id``, ``name``, ``date`` attributes).
        url_results: Triples ``(event, url_or_None, status)`` as
            produced by the per-event ``_enumerate_bfo_url`` loop.
        path: Output JSON path; parent dirs created if missing.
        batch_id: Batch identifier (typically
            ``datetime.now().strftime("%Y%m%d-%H%M%S")``).
    """
    n_in_scope = len(gap_events)
    # Plan 21-01 doesn't distinguish "no_odds" vs "partial_odds" at the
    # query level (the SQL uses NOT IN with closing_implied_prob IS NOT
    # NULL — events with ANY non-null odds row are excluded). The split
    # is reserved for Plan 21-02 finer-grained reporting.
    n_events_with_no_odds = n_in_scope
    n_events_with_partial_odds = 0

    n_ok = sum(1 for _e, _u, s in url_results if s == "OK")
    n_not_found = sum(1 for _e, _u, s in url_results if s == "EVENT_NOT_FOUND")
    n_disamb = sum(1 for _e, _u, s in url_results if s == "DISAMBIGUATION_FAIL")

    # Wall-clock estimate: BFO_DELAY_SECONDS per acquired URL + parse overhead.
    # Round up via int(... + 0.999) to avoid 0.5 → 0 rounding for small batches.
    est_seconds = n_ok * (BFO_DELAY_SECONDS + 0.3)  # +0.3s parse overhead
    est_minutes = max(1, int((est_seconds / 60.0) + 0.999))

    # Per-year breakdown: group url_results by event.date.year.
    per_year: dict[str, dict[str, int]] = {}
    for event, _url, status in url_results:
        year_key = str(event.date.year)
        bucket = per_year.setdefault(
            year_key,
            {
                "n_events": 0,
                "url_acquired": 0,
                "event_not_found": 0,
                "disambiguation_fail": 0,
            },
        )
        bucket["n_events"] += 1
        if status == "OK":
            bucket["url_acquired"] += 1
        elif status == "EVENT_NOT_FOUND":
            bucket["event_not_found"] += 1
        elif status == "DISAMBIGUATION_FAIL":
            bucket["disambiguation_fail"] += 1

    # Plan 21-02 forward-compat: persist the per-event URL list so
    # ``--scrape`` can read URLs without re-running the BFO /search
    # enumeration (which is the slow part of the dry-run, ~21 min for
    # ~1000 events). The current ``BFO_V22_GAP.json`` from Plan 21-01
    # Task 5 does NOT have this field — when ``--scrape`` runs against
    # the existing JSON it falls back to re-enumeration. New dry-runs
    # (post this refactor) include the list verbatim.
    url_acquired_list: list[dict[str, Any]] = []
    for event, url, status in url_results:
        if status == "OK" and url is not None:
            url_acquired_list.append(
                {
                    "event_id": event.id,
                    "event_name": event.name,
                    "event_date": event.date.isoformat(),
                    "url": url,
                }
            )

    payload: dict[str, Any] = {
        "batch_id": batch_id,
        "gap_query": {
            "cutoff_date": BFO_ARCHIVE_START_DATE.isoformat(),
            "n_in_scope_events": n_in_scope,
            "n_events_with_no_odds": n_events_with_no_odds,
            "n_events_with_partial_odds": n_events_with_partial_odds,
        },
        "url_resolution": {
            "url_acquired": n_ok,
            "event_not_found": n_not_found,
            "disambiguation_fail": n_disamb,
        },
        "estimated_scrape_wall_clock_min": est_minutes,
        "per_year_breakdown": per_year,
        "url_acquired_list": url_acquired_list,
        "next_step": (
            "Operator reviews this JSON, then approves Plan 21-02 Task 1 "
            "live scrape via BFOScraper.scrape_event_urls."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# ── Plan 21-02 Task 1: --scrape mode helpers ────────────────────────────


def _compute_xgb_v2_sha() -> str:
    """Return the SHA-256 hex digest of ``models/xgb_v2.joblib``."""
    h = hashlib.sha256()
    with Path(XGB_V2_MODEL_PATH).open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _emit_audit_01_mid_sha() -> None:
    """Write current xgb_v2 SHA to ``AUDIT_01_SHA_MID_PATH`` and gate.

    Per CONTEXT.md D-09(P15): the model file MUST remain byte-identical
    across Phase 21. ``--scrape`` is the mid-checkpoint of the AUDIT-01
    chain (Plan 21-02 Task 1 step 5); if the SHA has drifted from the
    canonical baseline, this raises ``RuntimeError`` so ``main`` can
    exit 1.

    Writes the SHA file unconditionally (so the operator has a paper
    trail even if the gate fails); raises only AFTER the file is on
    disk.
    """
    actual = _compute_xgb_v2_sha()
    sha_path = Path(AUDIT_01_SHA_MID_PATH)
    sha_path.parent.mkdir(parents=True, exist_ok=True)
    sha_path.write_text(actual + "\n", encoding="utf-8")

    baseline = Path(AUDIT_01_BASELINE_SHA_PATH).read_text(encoding="utf-8").strip()
    if actual != baseline:
        msg = (
            f"AUDIT-01 mid SHA mismatch: xgb_v2.joblib has drifted!\n"
            f"  expected (baseline): {baseline}\n"
            f"  actual:              {actual}\n"
            "  CONTEXT.md D-09(P15): xgb_v2 MUST remain byte-identical "
            "through Phase 21."
        )
        raise RuntimeError(msg)


def _load_gap_json_for_scrape(
    gap_json_path: Path,
    batch_id_arg: str,
) -> tuple[str, list[str]]:
    """Load BFO_V22_GAP.json + return (batch_id, url_list) after validation.

    Validation:
      1. The JSON's ``batch_id`` MUST equal ``batch_id_arg`` (prevents
         scraping against a stale gap report).
      2. If ``url_acquired_list`` is present (post-refactor schema): use
         it directly.
      3. Otherwise (legacy Plan 21-01 Task 5 JSON): raise
         ``RuntimeError`` with a clear message asking the operator to
         re-run --dry-run. We deliberately do NOT silently re-enumerate
         here because (a) the operator has not approved the time cost
         (~21 min) and (b) re-enumeration may yield a different URL set
         than the approved Task 5 audit trail.

    Returns:
        ``(batch_id, [url, url, ...])``
    """
    if not gap_json_path.exists():
        msg = f"BFO_V22_GAP.json not found at {gap_json_path}"
        raise FileNotFoundError(msg)

    payload = json.loads(gap_json_path.read_text(encoding="utf-8"))
    json_batch_id = payload.get("batch_id")
    if json_batch_id != batch_id_arg:
        msg = (
            f"--batch-id mismatch: argument {batch_id_arg!r} != "
            f"BFO_V22_GAP.json batch_id {json_batch_id!r}. Refusing to "
            "scrape against a different batch report. Either pass the "
            "matching --batch-id or re-run --dry-run to refresh the JSON."
        )
        raise ValueError(msg)

    url_list = payload.get("url_acquired_list")
    if not url_list:
        msg = (
            f"BFO_V22_GAP.json (batch_id={json_batch_id}) has no "
            "url_acquired_list field. Re-run `uv run python "
            "scripts/bfo_backfill.py --dry-run` to refresh the JSON with "
            "the per-event URL list (the dry-run is now schema-aware)."
        )
        raise RuntimeError(msg)

    # De-duplicate URLs while preserving first-seen order. Required because
    # the gap query can return multiple DB Event rows that resolve to the
    # same BFO URL (e.g. event_id 461 "UFC Event - 2007-06-12" and event_id
    # 1264 "UFC Fight Night: Stout vs Fisher" both resolve to
    # ufc-fight-night-ufc-fight-night-206-2412). Without dedup,
    # BFOScraper.scrape_event_urls hits the same URL N times and emits N
    # copies of identical rows. BFOOddsIngester's "expected 2 rows" defense
    # then skips the entire fight_id (rows_upserted=0 outcome). Discovered
    # 2026-05-15 during Plan 21-02 Task 3 first ingest attempt against
    # batch 20260515-001457 (861 URLs collapsed to 462 unique).
    #
    # WR-07 fix (REVIEW.md): the previous loop silently dropped entries
    # whose `url` field was None/empty. A JSON with ``url_acquired_list``
    # containing N entries but with every `url` field null would pass the
    # outer ``if not url_list`` check, then yield ``urls = []`` after the
    # dedup loop — and the --scrape mode would proceed with 0 URLs,
    # emitting an empty MANIFEST/CSV and exiting "successfully" while
    # nothing was actually scraped. Now: validate each entry has a
    # non-empty `url` field at load time AND raise if dedup yields 0
    # unique URLs.
    seen: set[str] = set()
    urls: list[str] = []
    for entry in url_list:
        u = entry.get("url")
        if not u:
            msg = (
                f"BFO_V22_GAP.json (batch_id={json_batch_id}) "
                f"url_acquired_list entry missing/empty 'url' field: "
                f"{entry!r}. Re-run --dry-run to refresh the JSON."
            )
            raise RuntimeError(msg)
        if u not in seen:
            seen.add(u)
            urls.append(u)
    if not urls:
        msg = (
            f"BFO_V22_GAP.json (batch_id={json_batch_id}) "
            "url_acquired_list yielded 0 unique URLs after dedup. "
            "Refusing to scrape an empty URL set; re-run --dry-run."
        )
        raise RuntimeError(msg)
    return (json_batch_id, urls)


def _write_manifest(
    *,
    batch_id: str,
    n_urls: int,
    n_events: int,
    n_rows: int,
    duration_seconds: float,
    captcha_hit_count: int,
    error_count: int,
    path: Path,
) -> None:
    """Write MANIFEST.json with the 8 fields per Plan 21-02 Task 1 step 4."""
    manifest = {
        "batch_id": batch_id,
        "scraped_at": datetime.now(UTC)
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        ),
        "n_urls": n_urls,
        "n_events": n_events,
        "n_rows": n_rows,
        "scrape_duration_seconds": round(duration_seconds, 3),
        "captcha_hit_count": captcha_hit_count,
        "error_count": error_count,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _write_fighters_names_csv_placeholder(path: Path) -> None:
    """Write the fighters_names.csv that Plan 21-02 Task 3 ingest expects.

    Per the prompt design: the existing ``BFOOddsIngester._resolve_fighter``
    handles fighter-name fuzzy matching at ingest time, so this file is a
    placeholder satisfying the Task 1 step 4 existence check. The header
    matches the ufcscraper / ``BFOFighterName`` schema so the existing
    ingester reader does not choke on the empty file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Header-only CSV. Comment row uses CSV `#` convention (which the
    # ingester reader treats as a regular row but we keep it column-shaped
    # so a future enhancement can fill in real fuzzy candidates).
    path.write_text(
        "fighter_id,database,name,database_id\n"
        "# Plan 21-02 Task 1: fighter-name resolution is delegated to "
        "BFOOddsIngester._resolve_fighter at ingest time.\n",
        encoding="utf-8",
    )


def _count_data_rows(csv_path: Path) -> int:
    """Return the number of non-header rows in the scraped CSV."""
    if not csv_path.exists():
        return 0
    with csv_path.open(encoding="utf-8") as fh:
        # Subtract the header row.
        return max(0, sum(1 for _ in fh) - 1)


def _count_unique_events(csv_path: Path) -> int:
    """Count distinct event-date prefixes in the composite fight_id column.

    Composite ``fight_id`` shape = ``"{date_iso}|{a_slug}|{b_slug}"`` (per
    BFOScraper.scrape_event_urls). The date prefix is unique per event,
    so counting distinct prefixes gives the number of events that
    contributed at least one row.

    WR-04 fix (REVIEW.md): the prefix MUST parse as ISO date, otherwise
    a malformed fight_id (e.g. ``"jones|cormier"`` with no date prefix)
    would silently inflate the event count via ``"jones"`` as a key.
    Malformed prefixes are now logged + skipped — the MANIFEST.json
    n_events telemetry stays trustworthy for operator coverage assessment.
    """
    if not csv_path.exists():
        return 0
    import csv as _csv

    events: set[str] = set()
    with csv_path.open(encoding="utf-8") as fh:
        reader = _csv.reader(fh)
        try:
            next(reader)  # skip header
        except StopIteration:
            return 0
        for row in reader:
            if not row:
                continue
            fight_id = row[0]
            prefix = fight_id.split("|", 1)[0]
            if not prefix:
                continue
            try:
                date.fromisoformat(prefix)
            except (ValueError, TypeError):
                logger.warning(
                    "[bfo-backfill] _count_unique_events: skipping "
                    "malformed fight_id (date prefix not ISO): %r",
                    fight_id,
                )
                continue
            events.add(prefix)
    return len(events)


def _run_scrape_mode(
    *,
    batch_id_arg: str,
    gap_json_path: Path,
) -> int:
    """Implement ``--scrape`` end-to-end. Returns exit code (0 = success)."""
    # 1. Load + validate GAP JSON.
    try:
        batch_id, urls = _load_gap_json_for_scrape(gap_json_path, batch_id_arg)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[bfo-backfill] ERROR: {exc}", file=sys.stderr)
        return 1

    n_urls = len(urls)
    print(f"[bfo-backfill] --scrape mode engaged. batch_id={batch_id} n_urls={n_urls}")

    # 2. Prepare batch dir.
    batch_dir = Path(SCRAPE_BATCH_DIR) / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    odds_csv_path = batch_dir / "BestFightOdds_odds.csv"
    names_csv_path = batch_dir / "fighters_names.csv"
    manifest_path = batch_dir / "MANIFEST.json"

    # 3. Lazy imports — keeps tests that monkeypatch sys.modules effective.
    from ufc_prediction.scraper.bfo_scraper import BFOScraper
    from ufc_prediction.scraper.client import ScraperClient

    client = ScraperClient(
        delay=BFO_DELAY_SECONDS,
        max_retries=BFO_MAX_RETRIES,
        timeout=BFO_TIMEOUT_SECONDS,
    )
    # Plan 21-02 Task 1: BFOScraper.__init__ requires (client, session,
    # data_folder, workers). For URL-list scrape we don't use session or
    # data_folder (scrape_event_urls writes to its own output_path), so
    # pass session=None + data_folder=batch_dir for symmetry.
    #
    # CR-02 fix (REVIEW.md): the previous `except TypeError → BFOScraper(client)`
    # fallback was dead code (test fakes use ``def __init__(self, *args,
    # **kwargs)`` so they accept the 4-arg call uniformly) AND it silently
    # masked legitimate TypeErrors from future signature drift or real
    # constructor bugs (e.g. ``Path(data_folder)`` failing when None). If a
    # genuine TypeError surfaces here, it should propagate — not be swallowed
    # into a permissive 1-arg construction path that hides the bug.
    scraper = BFOScraper(
        client=client,
        session=None,
        data_folder=batch_dir,
    )

    # 4. Scrape — count captcha hits + errors via logging-record capture.
    captcha_hit_count = 0
    error_count = 0

    bfo_logger = logging.getLogger("ufc_prediction.scraper.bfo_scraper")

    class _StatsHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            nonlocal captcha_hit_count, error_count
            msg = record.getMessage().lower()
            if "captcha" in msg:
                captcha_hit_count += 1
            elif "fetch failed" in msg or "parse failed" in msg or "returned none" in msg:
                error_count += 1

    stats_handler = _StatsHandler(level=logging.WARNING)
    bfo_logger.addHandler(stats_handler)

    t_start = time.monotonic()
    try:
        scraper.scrape_event_urls(urls, output_path=odds_csv_path)
    finally:
        bfo_logger.removeHandler(stats_handler)
    duration = time.monotonic() - t_start

    # 5. Tally output.
    n_rows = _count_data_rows(odds_csv_path)
    n_events = _count_unique_events(odds_csv_path)

    # 6. Write fighters_names.csv placeholder.
    _write_fighters_names_csv_placeholder(names_csv_path)

    # 7. Write MANIFEST.json.
    _write_manifest(
        batch_id=batch_id,
        n_urls=n_urls,
        n_events=n_events,
        n_rows=n_rows,
        duration_seconds=duration,
        captcha_hit_count=captcha_hit_count,
        error_count=error_count,
        path=manifest_path,
    )

    # 8. AUDIT-01 mid SHA gate (Task 1 step 5). HARD gate — exit 1 on drift.
    try:
        _emit_audit_01_mid_sha()
    except RuntimeError as exc:
        print(f"[bfo-backfill] {exc}", file=sys.stderr)
        return 1

    # 9. 5-line summary.
    print(
        f"[bfo-backfill] --scrape complete:\n"
        f"[bfo-backfill]   n_urls                  = {n_urls}\n"
        f"[bfo-backfill]   n_rows                  = {n_rows}\n"
        f"[bfo-backfill]   scrape_duration_seconds = {duration:.1f}\n"
        f"[bfo-backfill]   captcha_hit_count       = {captcha_hit_count}\n"
        f"[bfo-backfill]   error_count             = {error_count}\n"
        f"[bfo-backfill]   batch_dir               = {batch_dir}\n"
        f"[bfo-backfill]   AUDIT-01 mid SHA        = {AUDIT_01_SHA_MID_PATH} "
        "(byte-identical to baseline)"
    )
    return 0


# ── Plan 21-02 Task 2: --integrity-check mode (pre-ingest gate) ─────────


class BFOIntegrityError(Exception):
    """Pre-ingest rematch-integrity violation; halt before BFOOddsIngester.ingest_all().

    Raised by ``_run_integrity_check`` when one or more (pair_key,
    event_date) groups contain ≥2 distinct ``fight_id`` values — the
    999.4-class regression where ``pk_fight_odds`` upserts would silently
    overwrite each other. The ``violations`` attribute is the source of
    truth for the operator-action checkpoint (CONTEXT.md D-05a).
    """

    def __init__(self, violations: list[dict[str, Any]]):
        self.violations = violations
        super().__init__(
            f"BFO rematch integrity violation: {len(violations)} pair(s) failed distinctness check"
        )


def _parse_pair_key_from_fight_id(fight_id: str) -> tuple[str, str]:
    """Extract canonical (pair-symmetric) pair_key from composite ``fight_id``.

    Composite shape: ``"{date_iso}|{a_slug}|{b_slug}"`` (mirrors
    ``BFOScraper.scrape_event_urls`` and ``BFOOddsRow.event_date``). The
    pair_key is ``tuple(sorted([a_slug, b_slug]))`` so the same pair
    yields an identical key regardless of which fighter is "a" vs "b" in
    the BFO row ordering.

    Raises:
        ValueError: ``fight_id`` doesn't have exactly 3 ``|``-separated
        segments. Caught + surfaced by ``_run_integrity_check`` as a
        per-row warning (we don't halt the whole gate on a single
        malformed row; that's the 999.4 raise's job).
    """
    parts = fight_id.split("|")
    if len(parts) != 3:
        raise ValueError(
            f"BFO fight_id has unexpected segment count: {fight_id!r} "
            f"(expected 'date|a|b', got {len(parts)} segments)"
        )
    _date, a, b = parts
    if not a or not b:
        raise ValueError(f"BFO fight_id has empty fighter segment: {fight_id!r}")
    # WR-02 fix (REVIEW.md): build an explicit 2-tuple from the sorted
    # pair so the return type matches the annotation ``tuple[str, str]``
    # without needing a ``# type: ignore[return-value]`` (which previously
    # masked the fact that ``tuple(sorted([a, b]))`` has type
    # ``tuple[str, ...]``).
    pair = sorted([a, b])
    return (pair[0], pair[1])


def _run_integrity_check(csv_path: Path) -> dict[str, Any]:
    """Pre-ingest rematch-integrity gate against a BestFightOdds_odds.csv.

    Per CONTEXT.md D-05 (REVISED): Python-level check on the new CSV
    before passing to ``BFOOddsIngester.ingest_all()``. Same semantic
    guarantee as the SQL check would have provided against a staging
    table — every (pair_key, event_date) tuple must map to AT MOST one
    distinct ``fight_id``. Violation = the 999.4 regression class where
    ``pk_fight_odds`` upserts silently overwrite each other.

    Workflow:
      1. Validate the CSV exists (raise FileNotFoundError if not).
      2. Iterate rows via the same ``BFOOddsRow`` Pydantic validator
         the production ingester uses (``_load_odds_rows`` pattern).
         Malformed rows are warned + skipped (matches ingester behavior).
      3. Group rows by canonical pair_key parsed from ``fight_id``.
      4. For every pair_key with ≥2 distinct fight_ids: assert that the
         set of distinct event_dates equals the set of distinct fight_ids
         (i.e., each rematch has its OWN event_date). Any same-date
         distinct-fight_id pair is a violation.
      5. On clean: return summary dict; on violation: raise
         ``BFOIntegrityError`` with the per-pair violation details.

    Args:
        csv_path: Absolute (or cwd-relative) path to
            ``data/bfo/v22-backfill/<batch_id>/BestFightOdds_odds.csv``.

    Returns:
        ``{"n_pairs": N, "n_rematches": M, "n_rows": K, "violations": []}``
        where ``n_rematches`` counts pair_keys with ≥2 distinct fight_ids.

    Raises:
        FileNotFoundError: ``csv_path`` does not exist on disk.
        BFOIntegrityError: ≥1 (pair_key, event_date) group has multiple
            distinct fight_ids.
    """
    if not csv_path.exists():
        msg = (
            f"BFO integrity-check input CSV not found: {csv_path}. "
            "Run --scrape first or pass --batch-id matching an existing "
            "data/bfo/v22-backfill/<batch_id>/ directory."
        )
        raise FileNotFoundError(msg)

    # Lazy import — keeps tests that don't touch real CSVs from paying the
    # Pydantic import cost, and mirrors the lazy-import pattern used by
    # _query_gap_events / _run_scrape_mode elsewhere in this module.
    import csv as _csv

    from ufc_prediction.scraper.bfo_models import BFOOddsRow

    # pair_key → event_date → set of distinct fight_ids seen.
    by_pair: dict[
        tuple[str, str],
        dict[Any, set[str]],
    ] = {}
    n_rows = 0

    with csv_path.open(encoding="utf-8") as fh:
        for raw in _csv.DictReader(fh):
            try:
                odds_row = BFOOddsRow(**raw)
            except Exception as exc:
                logger.warning(
                    "[bfo-backfill] integrity-check: invalid row in %s: %r (%s)",
                    csv_path,
                    raw,
                    exc,
                )
                continue
            try:
                pair_key = _parse_pair_key_from_fight_id(odds_row.fight_id)
                event_date = odds_row.event_date
            except ValueError as exc:
                logger.warning(
                    "[bfo-backfill] integrity-check: skipping row with bad fight_id %r: %s",
                    odds_row.fight_id,
                    exc,
                )
                continue
            n_rows += 1
            by_date = by_pair.setdefault(pair_key, {})
            by_date.setdefault(event_date, set()).add(odds_row.fight_id)

    # Tally + violation detection.
    violations: list[dict[str, Any]] = []
    n_pairs = len(by_pair)
    n_rematches = 0
    for pair_key, by_date in by_pair.items():
        all_fight_ids = {fid for fids in by_date.values() for fid in fids}
        if len(all_fight_ids) >= 2:
            n_rematches += 1
        # Per-event-date distinctness: each event_date must have exactly
        # one fight_id; multiple distinct fight_ids on the same date for
        # the same pair = the 999.4 regression class.
        for event_date, fight_ids in by_date.items():
            if len(fight_ids) >= 2:
                violations.append(
                    {
                        "pair_key": pair_key,
                        "event_date": event_date.isoformat(),
                        "fight_ids": sorted(fight_ids),
                    }
                )

    if violations:
        # Print violators to stderr so they surface even when the caller
        # swallows the exception's message.
        for v in violations:
            print(
                f"[bfo-backfill] INTEGRITY VIOLATION: pair={v['pair_key']} "
                f"event_date={v['event_date']} fight_ids={v['fight_ids']}",
                file=sys.stderr,
            )
        raise BFOIntegrityError(violations=violations)

    return {
        "n_pairs": n_pairs,
        "n_rematches": n_rematches,
        "n_rows": n_rows,
        "violations": [],
    }


# ── Plan 21-02 Task 4: --emit-coverage mode (post-ingest delta JSON) ────


def _coverage_before_from_gap_json(payload: dict[str, Any]) -> float:
    """Derive ``coverage_before`` from the BFO_V22_GAP.json payload.

    Formula choice: ``coverage_before`` = 1 - (gap_events / in_scope_events)
    where ``gap_events`` = ``n_events_with_no_odds + n_events_with_partial_odds``
    and ``in_scope_events`` = ``n_in_scope_events``. This is an event-level
    proxy for the fight-level coverage_after metric (the dry-run gap query
    is event-keyed, not fight-keyed; the conversion factor cancels out as
    long as fights/event is roughly constant across the gap, which holds at
    cutoff-date discipline).

    Returns 0.0 when ``n_in_scope_events == 0`` (degenerate empty-gap case).
    """
    gq = payload.get("gap_query") or {}
    in_scope = int(gq.get("n_in_scope_events", 0) or 0)
    if in_scope == 0:
        return 0.0
    n_no_odds = int(gq.get("n_events_with_no_odds", 0) or 0)
    n_partial = int(gq.get("n_events_with_partial_odds", 0) or 0)
    gap = n_no_odds + n_partial
    return round(1.0 - (gap / in_scope), 4)


def _query_coverage_after(session: Any) -> tuple[int, int]:
    """Return ``(covered, total)`` post-2007 fight counts via SQLAlchemy.

    Numerator: COUNT(DISTINCT fight.id) WHERE event.date >= 2007-06-01
        AND fight_odds.closing_implied_prob IS NOT NULL
    Denominator: COUNT(DISTINCT fight.id) WHERE event.date >= 2007-06-01
    """
    from sqlalchemy import func, select

    from ufc_prediction.models.event import Event
    from ufc_prediction.models.fight import Fight
    from ufc_prediction.models.fight_odds import FightOdds

    # Denominator: distinct in-scope fight ids.
    total_stmt = (
        select(func.count(func.distinct(Fight.id)))
        .join(Event, Event.id == Fight.event_id)
        .where(Event.date >= BFO_ARCHIVE_START_DATE)
    )
    # Numerator: distinct in-scope fight ids that have a non-null
    # closing_implied_prob row in fight_odds.
    covered_stmt = (
        select(func.count(func.distinct(Fight.id)))
        .join(Event, Event.id == Fight.event_id)
        .join(FightOdds, FightOdds.fight_id == Fight.id)
        .where(Event.date >= BFO_ARCHIVE_START_DATE)
        .where(FightOdds.closing_implied_prob.is_not(None))
    )
    # Single combined SELECT keeps the test mock simple: one .execute()
    # call, one (covered, total) tuple back. Production would issue 2
    # separate queries; the mock pattern in tests batches them into one
    # tuple per call. We use the combined-tuple form for symmetry.
    combined_stmt = select(
        covered_stmt.scalar_subquery(),
        total_stmt.scalar_subquery(),
    )
    return tuple(session.execute(combined_stmt).one())  # type: ignore[return-value]


def _query_per_year_coverage(
    session: Any,
    years: list[int],
) -> dict[str, tuple[int, int]]:
    """Per-year ``{year_str: (covered, total)}`` using the same SQL shape."""
    from sqlalchemy import extract, func, select

    from ufc_prediction.models.event import Event
    from ufc_prediction.models.fight import Fight
    from ufc_prediction.models.fight_odds import FightOdds

    out: dict[str, tuple[int, int]] = {}
    for year in years:
        total_stmt = (
            select(func.count(func.distinct(Fight.id)))
            .join(Event, Event.id == Fight.event_id)
            .where(extract("year", Event.date) == year)
        )
        covered_stmt = (
            select(func.count(func.distinct(Fight.id)))
            .join(Event, Event.id == Fight.event_id)
            .join(FightOdds, FightOdds.fight_id == Fight.id)
            .where(extract("year", Event.date) == year)
            .where(FightOdds.closing_implied_prob.is_not(None))
        )
        combined_stmt = select(
            covered_stmt.scalar_subquery(),
            total_stmt.scalar_subquery(),
        )
        c, t = tuple(session.execute(combined_stmt).one())
        out[str(year)] = (int(c or 0), int(t or 0))
    return out


def _emit_coverage_json(
    *,
    batch_id: str,
    ingest_summary: Any,
    gap_json_path: Path,
    coverage_json_path: Path,
    session: Any,
    n_train_pre: int,
    n_train_post: int,
    n_test_pre: int,
    n_test_post: int,
    year_range: tuple[int, int] = (2007, 2026),
) -> dict[str, Any]:
    """Compose + write the BFO_V22_COVERAGE.json artifact (Plan 21-02 Task 4).

    Args:
        batch_id: Batch identifier (e.g. ``"20260514-231731"``); round-trips
            into the JSON for operator + AUDIT-01 traceability.
        ingest_summary: ``BFOOddsIngester.ingest_all()`` return value
            (``IngestSummary`` dataclass) OR a dict with the same keys.
            Used today only for the operator-readable ``known_acquisition_gaps``
            note; the coverage numerics come from the DB query.
        gap_json_path: Path to BFO_V22_GAP.json (Plan 21-01 output) — read
            for ``coverage_before``.
        coverage_json_path: Output path for BFO_V22_COVERAGE.json.
        session: SQLAlchemy session for the coverage_after queries.
            Injected for testability (production: SessionLocal()).
        n_train_pre: Pre-ingest training-set fight count (operator-supplied
            via --n-train-pre). BFO-V22-04 cutoff-date discipline check.
        n_train_post: Post-ingest training-set fight count (operator).
        n_test_pre: Pre-ingest test-set fight count (operator).
        n_test_post: Post-ingest test-set fight count (operator).
        year_range: ``(start_year_inclusive, end_year_exclusive)`` for the
            per-year breakdown loop. Default 2007-2026 covers the full BFO
            archive scope; tests can pass a narrower range to cap query
            count.

    Returns:
        The same dict written to disk (round-trip-equivalent JSON).
    """
    # 1. Read GAP JSON → coverage_before.
    gap_payload = json.loads(gap_json_path.read_text(encoding="utf-8"))
    coverage_before = _coverage_before_from_gap_json(gap_payload)

    # 2. DB query: coverage_after.
    covered, total = _query_coverage_after(session)
    coverage_after = round(covered / total, 4) if total > 0 else 0.0

    # 3. Per-year breakdown.
    years = list(range(year_range[0], year_range[1]))
    per_year_raw = _query_per_year_coverage(session, years)
    per_year_breakdown: dict[str, dict[str, float]] = {}
    for year_str, (c, t) in per_year_raw.items():
        if t == 0:
            continue  # skip year-buckets with no in-scope fights
        after_y = round(c / t, 4)
        # We don't have per-year before-coverage in the GAP JSON; report
        # the after-only value with a None placeholder for delta. Operator
        # can cross-reference the GAP JSON's per_year_breakdown field
        # directly.
        per_year_breakdown[year_str] = {
            "covered": c,
            "total": t,
            "after": after_y,
        }

    # 4. Cutoff-date discipline (BFO-V22-04 hard check).
    discipline_preserved = n_train_pre == n_train_post and n_test_pre == n_test_post

    # 5. Known acquisition gaps. The GAP JSON's url_resolution.disambiguation_fail
    # count is the BFO-side acquisition gap (events the /search couldn't
    # disambiguate); ingest_summary.unmatched_bfo_names captures the
    # ingest-side fighter-resolution gap. We surface the disambiguation
    # count as the primary acquisition-gap signal for the operator.
    url_res = gap_payload.get("url_resolution") or {}
    n_disamb = int(url_res.get("disambiguation_fail", 0) or 0)
    n_not_found = int(url_res.get("event_not_found", 0) or 0)
    known_gaps: list[dict[str, Any]] = []
    if n_disamb > 0:
        known_gaps.append(
            {
                "category": "DISAMBIGUATION_FAIL",
                "count": n_disamb,
                "note": (
                    "BFO /search returned a non-ufc-prefixed slug "
                    "(Bellator/Strikeforce collision). See BFO_V22_GAP.json "
                    "per_year_breakdown for year-level distribution."
                ),
            }
        )
    if n_not_found > 0:
        known_gaps.append(
            {
                "category": "EVENT_NOT_FOUND",
                "count": n_not_found,
                "note": "BFO /search returned no matching event.",
            }
        )

    # 6. Compose payload.
    payload: dict[str, Any] = {
        "batch_id": batch_id,
        "ingested_at": datetime.now(UTC)
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        ),
        "coverage_before": coverage_before,
        "coverage_after": coverage_after,
        "coverage_delta": round(coverage_after - coverage_before, 4),
        "target_rate_threshold": TARGET_RATE_THRESHOLD,
        "target_rate_achieved": coverage_after >= TARGET_RATE_THRESHOLD,
        "n_training_fights_pre": n_train_pre,
        "n_training_fights_post": n_train_post,
        "n_test_fights_pre": n_test_pre,
        "n_test_fights_post": n_test_post,
        "cutoff_date_discipline_preserved": discipline_preserved,
        "per_year_coverage_breakdown": per_year_breakdown,
        "known_acquisition_gaps": known_gaps,
    }

    # 7. Persist + return.
    coverage_json_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _run_emit_coverage_mode(
    *,
    batch_id: str,
    gap_json_path: Path,
    n_train_pre: int,
    n_train_post: int,
    n_test_pre: int,
    n_test_post: int,
) -> int:
    """End-to-end --emit-coverage driver. Returns exit code (0 = success).

    Always exits 0 when the JSON writes successfully. The
    target_rate_achieved boolean is the operator-decision-point signal —
    on False, a stderr warning surfaces but the mode does NOT halt
    (per the prompt: the JSON IS the artifact; the operator decides
    next steps based on it).
    """
    # Lazy import — keeps tests that monkeypatch sys.modules effective.
    from ufc_prediction.db.session import SessionLocal

    session = SessionLocal()
    try:
        result = _emit_coverage_json(
            batch_id=batch_id,
            ingest_summary={},  # Plan 21-02 Task 4: operator-supplied numerics
            gap_json_path=gap_json_path,
            coverage_json_path=Path(COVERAGE_JSON_PATH),
            session=session,
            n_train_pre=n_train_pre,
            n_train_post=n_train_post,
            n_test_pre=n_test_pre,
            n_test_post=n_test_post,
        )
    finally:
        session.close()

    # 5-line operator summary.
    print(
        f"[bfo-backfill] --emit-coverage complete:\n"
        f"[bfo-backfill]   coverage_before                  = "
        f"{result['coverage_before']}\n"
        f"[bfo-backfill]   coverage_after                   = "
        f"{result['coverage_after']}\n"
        f"[bfo-backfill]   coverage_delta                   = "
        f"{result['coverage_delta']}\n"
        f"[bfo-backfill]   target_rate_achieved             = "
        f"{result['target_rate_achieved']}\n"
        f"[bfo-backfill]   cutoff_date_discipline_preserved = "
        f"{result['cutoff_date_discipline_preserved']}"
    )

    if not result["target_rate_achieved"]:
        print(
            f"[bfo-backfill] WARNING: coverage_after "
            f"({result['coverage_after']}) below target "
            f"({result['target_rate_threshold']}). Operator-decision-point: "
            "accept partial coverage OR extend phase with retry pass.",
            file=sys.stderr,
        )
    return 0


# ── argparse + main() ───────────────────────────────────────────────────


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser (testable hook).

    Plan 21-01 ships ``--dry-run`` (gap query + URL resolution only;
    emits ``BFO_V22_GAP.json``) and ``--resume`` (flag stub; full logic
    in Plan 21-02 batch-resume implementation).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Phase 21 BFO Coverage Backfill driver — Plan 21-01 scaffolding "
            "(gap query + URL enumeration + dry-run gap-report emit)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip live scraping; perform gap query + URL resolution and "
            "emit BFO_V22_GAP.json for operator review (Task 5 checkpoint)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "[STUB — Plan 21-02] Resume from a partial batch by reading "
            "data/bfo/v22-backfill/<batch_id>/MANIFEST.json. Flag is "
            "exposed in Plan 21-01 for argparser stability; full logic "
            "lands in Plan 21-02 Task 1."
        ),
    )
    parser.add_argument(
        "--gap-json-path",
        default=GAP_JSON_PATH,
        help=f"Output path for BFO_V22_GAP.json (default: {GAP_JSON_PATH})",
    )
    # Plan 21-02 Task 1: live scrape mode.
    parser.add_argument(
        "--scrape",
        action="store_true",
        help=(
            "Live BFO scrape against url_acquired URLs from BFO_V22_GAP.json. "
            "Requires --batch-id matching the JSON's batch_id. Writes "
            "data/bfo/v22-backfill/<batch_id>/{BestFightOdds_odds.csv, "
            "fighters_names.csv, MANIFEST.json} + AUDIT-01 mid SHA file."
        ),
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        help=(
            "Required when --scrape or --integrity-check: must match "
            "BFO_V22_GAP.json's batch_id (prevents accidentally scraping or "
            "checking against a stale gap report)."
        ),
    )
    # Plan 21-02 Task 2: pre-ingest rematch-integrity gate (CONTEXT.md D-05).
    parser.add_argument(
        "--integrity-check",
        action="store_true",
        help=(
            "Run pre-ingest rematch-integrity check on "
            "data/bfo/v22-backfill/<batch_id>/BestFightOdds_odds.csv. "
            "Requires --batch-id. Halts (exit 1) on violation; exits 0 "
            "with a summary line on clean. CONTEXT.md D-05 (REVISED) gate."
        ),
    )
    # Plan 21-02 Task 4: post-ingest coverage delta JSON.
    parser.add_argument(
        "--emit-coverage",
        action="store_true",
        help=(
            "Emit BFO_V22_COVERAGE.json (pre/post coverage delta + cutoff-"
            "date discipline check). Requires --batch-id + 4 count flags "
            "(--n-train-pre/--n-train-post/--n-test-pre/--n-test-post) "
            "captured by the operator before + after Task 3 ingest. "
            "Mutually-exclusive with --scrape, --dry-run, --integrity-check."
        ),
    )
    parser.add_argument(
        "--n-train-pre",
        type=int,
        default=None,
        help="Pre-ingest training-set fight count (BFO-V22-04 cutoff check).",
    )
    parser.add_argument(
        "--n-train-post",
        type=int,
        default=None,
        help="Post-ingest training-set fight count (BFO-V22-04 cutoff check).",
    )
    parser.add_argument(
        "--n-test-pre",
        type=int,
        default=None,
        help="Pre-ingest test-set fight count (BFO-V22-04 cutoff check).",
    )
    parser.add_argument(
        "--n-test-post",
        type=int,
        default=None,
        help="Post-ingest test-set fight count (BFO-V22-04 cutoff check).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Plan 21-01 entry point — gap query + URL resolution + JSON emit.

    Plan 21-01 scope: dry-run path only. Live scraping (--no --dry-run)
    is reserved for Plan 21-02 Task 1 — invoking main() without
    --dry-run prints a placeholder and exits 0 (the live path is added
    in Plan 21-02).
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[bfo-backfill] %(message)s")

    # AF startup asserts FIRST.
    _run_af_startup_asserts()
    print(
        "[bfo-backfill] AF OK: BFO_ARCHIVE_START_DATE=2007-06-01, "
        "TARGET_RATE_THRESHOLD=0.96, "
        "DISAMBIGUATION_REQUIRE_UFC_PREFIX=True"
    )

    if args.resume:
        print(
            "[bfo-backfill] --resume is a Plan 21-01 stub; Plan 21-02 "
            "Task 1 implements the resume logic. Exiting 0.",
        )
        return 0

    if args.integrity_check:
        # Plan 21-02 Task 2: pre-ingest rematch-integrity gate.
        if not args.batch_id:
            print(
                "[bfo-backfill] ERROR: --integrity-check requires --batch-id "
                "(used to locate "
                "data/bfo/v22-backfill/<batch_id>/BestFightOdds_odds.csv).",
                file=sys.stderr,
            )
            return 2
        csv_path = Path(SCRAPE_BATCH_DIR) / args.batch_id / "BestFightOdds_odds.csv"
        try:
            summary = _run_integrity_check(csv_path)
        except FileNotFoundError as exc:
            print(f"[bfo-backfill] ERROR: {exc}", file=sys.stderr)
            return 1
        except BFOIntegrityError as exc:
            # _run_integrity_check ALSO prints the per-violation lines to
            # stderr, but main() re-prints them here so callers that mock
            # _run_integrity_check still surface the violations to the
            # operator-action checkpoint (CONTEXT.md D-05a).
            for v in exc.violations:
                print(
                    "[bfo-backfill] INTEGRITY VIOLATION: "
                    f"pair={v.get('pair_key')} "
                    f"event_date={v.get('event_date')} "
                    f"fight_ids={v.get('fight_ids')}",
                    file=sys.stderr,
                )
            print(
                f"[bfo-backfill] {exc} — operator-action checkpoint "
                "required (re-scrape, manual fix, or abort).",
                file=sys.stderr,
            )
            return 1
        print(
            f"[bfo-backfill] Integrity check passed: "
            f"{summary['n_rematches']} rematches verified across "
            f"{summary['n_pairs']} pairs ({summary['n_rows']} rows total)"
        )
        return 0

    if args.emit_coverage:
        # Plan 21-02 Task 4: --emit-coverage requires --batch-id +
        # --n-train-pre/post + --n-test-pre/post (operator-captured before
        # + after Task 3 ingest, used for the BFO-V22-04 cutoff-date
        # discipline hard check).
        if not args.batch_id:
            print(
                "[bfo-backfill] ERROR: --emit-coverage requires --batch-id "
                "(round-trips into BFO_V22_COVERAGE.json for traceability).",
                file=sys.stderr,
            )
            return 2
        missing_count_args = [
            flag
            for flag, val in (
                ("--n-train-pre", args.n_train_pre),
                ("--n-train-post", args.n_train_post),
                ("--n-test-pre", args.n_test_pre),
                ("--n-test-post", args.n_test_post),
            )
            if val is None
        ]
        if missing_count_args:
            print(
                "[bfo-backfill] ERROR: --emit-coverage requires the "
                f"following count args: {', '.join(missing_count_args)} "
                "(operator captures these via "
                "`get_train_test_counts()` before + after Task 3 ingest; "
                "BFO-V22-04 cutoff-date discipline hard check).",
                file=sys.stderr,
            )
            return 2
        return _run_emit_coverage_mode(
            batch_id=args.batch_id,
            gap_json_path=Path(args.gap_json_path),
            n_train_pre=args.n_train_pre,
            n_train_post=args.n_train_post,
            n_test_pre=args.n_test_pre,
            n_test_post=args.n_test_post,
        )

    if args.scrape:
        # Plan 21-02 Task 1: --scrape requires --batch-id (must match the
        # GAP JSON's batch_id). argparse exits 2 on missing required args;
        # we use exit 1 here for the semantic-validation failure to keep
        # "argparse error" (2) distinct from "scrape input invalid" (1).
        if not args.batch_id:
            print(
                "[bfo-backfill] ERROR: --scrape requires --batch-id "
                "(must match BFO_V22_GAP.json's batch_id field).",
                file=sys.stderr,
            )
            return 2
        return _run_scrape_mode(
            batch_id_arg=args.batch_id,
            gap_json_path=Path(args.gap_json_path),
        )

    # Lazy DB + client imports — keeps test environments without the
    # full DB stack from breaking on module import.
    from ufc_prediction.db.session import SessionLocal
    from ufc_prediction.scraper.client import ScraperClient

    print("[bfo-backfill] Querying gap events from DB...")
    session = SessionLocal()
    try:
        gap_events = _query_gap_events(session)
    finally:
        session.close()
    print(
        f"[bfo-backfill] Gap query returned {len(gap_events)} events "
        f"(cutoff {BFO_ARCHIVE_START_DATE.isoformat()})."
    )

    if not gap_events:
        print(
            "[bfo-backfill] No gap events found — coverage may already "
            "be at target. Emitting empty BFO_V22_GAP.json."
        )

    # URL enumeration loop.
    client = ScraperClient(
        delay=BFO_DELAY_SECONDS,
        max_retries=BFO_MAX_RETRIES,
        timeout=BFO_TIMEOUT_SECONDS,
    )
    url_results: list[tuple[Any, str | None, str]] = []
    print(
        f"[bfo-backfill] Enumerating BFO URLs for {len(gap_events)} events "
        f"(rate-limited @ {BFO_DELAY_SECONDS}s/req)..."
    )
    for i, event in enumerate(gap_events, start=1):
        url, status = _enumerate_bfo_url(client, event.name)
        url_results.append((event, url, status))
        if i % 50 == 0:
            print(f"[bfo-backfill]   ...{i}/{len(gap_events)} events probed.")

    # Build batch_id + ensure batch dir exists for Plan 21-02 reuse.
    batch_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    batch_dir = Path(SCRAPE_BATCH_DIR) / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Emit gap report.
    gap_json_path = Path(args.gap_json_path)
    _emit_gap_json(
        gap_events=gap_events,
        url_results=url_results,
        path=gap_json_path,
        batch_id=batch_id,
    )

    n_ok = sum(1 for _e, _u, s in url_results if s == "OK")
    n_not_found = sum(1 for _e, _u, s in url_results if s == "EVENT_NOT_FOUND")
    n_disamb = sum(1 for _e, _u, s in url_results if s == "DISAMBIGUATION_FAIL")
    print(
        f"\n[bfo-backfill] Backfill scaffolding complete. Wrote:\n"
        f"  {gap_json_path}\n"
        f"[bfo-backfill] gap_events             = {len(gap_events)}\n"
        f"[bfo-backfill] url_acquired           = {n_ok}\n"
        f"[bfo-backfill] event_not_found        = {n_not_found}\n"
        f"[bfo-backfill] disambiguation_fail    = {n_disamb}\n"
        f"[bfo-backfill] batch_id               = {batch_id}\n"
        f"[bfo-backfill] batch_dir              = {batch_dir}\n"
        f"[bfo-backfill] next_step              = "
        f"operator reviews BFO_V22_GAP.json + approves Plan 21-02 Task 1"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
