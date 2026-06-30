#!/usr/bin/env python
"""Phase 20 / BFO-V22-00 — Archive reachability probe across 6 year-anchors.

Operator-driven entry point for the 1-day BFO reachability audit spike.
Mechanically pre-registers the post-2007 BFO coverage target consumed by
Phase 21 BFO backfill (BFO-V22-03 Wave-0 RED test) via a 30-request
probe across 6 year-anchors (2007, 2010, 2013, 2016, 2019, 2022) with
early-stop on consecutive non-reachable rows. Discriminates reachable
event pages from homepage-fallback HTTP 200s (Task 1 finding) AND from
Cloudflare CAPTCHA gates (Pitfall #4) using page-content inspection
rather than raw status codes.

Locked decisions (CONTEXT.md):
  - D-07: 6 anchor years × 5 events = 30 probe budget; early-stop after 5
    consecutive non-reachable rows per year. Operator-curated URLs in
    ``data/bfo_event_probe_targets.csv`` (Open Q #2 Option B; the
    ``Event`` schema has no BFO slug/id, so URL acquisition is the
    operator's responsibility — the audit MEASURES, it does not derive).
  - D-08: ``ScraperClient(delay=1.2, max_retries=3, timeout=5.0)`` — 5s
    timeout (NOT the 30s ScraperClient default); 1.2s delay; raw
    ``ScraperClient.get()`` per Anti-pattern #3 (do NOT call the BFO
    orchestrator's ``scrape_all`` method — its ``parse_errors`` counter
    will trip on event-page HTML structure, which differs from
    fighter-page schema).
  - D-09: Silva–Franklin I (UFC 64, 2006-10-14, pre-archive) → II
    (UFC 77, 2007-10-20, in-archive) is the canonical 2007-boundary
    fixture. The pre-2007 Couture–Liddell trilogy is structurally
    unreachable (BFO archive starts 2007 per STACK.md WebFetch).
  - D-10: 6 MD sections in order — Probe Methodology / Per-Year
    Reachability Table / URL-Format-Drift Findings / Pre-Registered
    Post-2007 Coverage Target / Couture–Liddell Substitution Note /
    Risk Flags.

Refinements folded in from Plan 20-02 Task 1 (operator CSV curation):
  - BFO returns HTTP 200 with the GENERIC HOMEPAGE
    (``<title>UFC &amp; MMA Odds &amp; Betting Lines | Best Fight Odds</title>``)
    for invalid event slugs. The probe MUST inspect the page <title> to
    distinguish a reachable event page from a homepage fallback.
  - Operator-marked unfindable rows use ``event_url == "EVENT_NOT_FOUND"``
    sentinel; these are recorded as ``status="event_not_found"`` without
    issuing an HTTP request (and count as non-reachable in totals).
  - The pre-registered post-2007 coverage target floors at 80% per the
    Phase 21 BFO-V22-03 expectations:
    ``target_rate = max(0.80, n_reachable_post2007 / n_total_post2007)``
    (excludes the 2007 boundary year from the denominator).

Anti-features (BANNED):
  - AF-1: hyperparameter retuning. This is an audit, not a model retrain;
    no ML touches.
  - AF-2: feature engineering. No new feature columns produced.
  - AF-3: model file mutation. ``models/xgb_v2.joblib`` is byte-identical
    (read-only); AUDIT-01 chain verifies SHA at phase end.
  - AF-4: orchestrator method invocation (Anti-pattern #3) — the audit
    uses raw ``ScraperClient.get()`` so event-page HTML never enters the
    fighter-page parser; the BFO scrape orchestrator is intentionally
    NOT imported.

Pitfalls mitigated:
  - Pitfall #3 (no derivable BFO slug from ``Event`` schema): operator
    pre-curates ``data/bfo_event_probe_targets.csv``; the audit script
    fails fast if any anchor year has < EVENTS_PER_YEAR rows.
  - Pitfall #4 (Cloudflare gate returns HTTP 200 with challenge body):
    every HTTP 200 is parsed with BeautifulSoup and checked via
    ``_check_captcha(soup)`` from ``bfo_scraper.py:191``; on
    ``BFOParseError`` the row is recorded as ``status="captcha"``,
    ``captcha_hit=True`` (NOT counted as reachable).
  - Pitfall #5 (denominator semantics): the post-2007 reachability rate
    is computed as ``n_reachable / n_total`` per year — homepage-fallback,
    not-found, captcha, error, and event_not_found rows all count toward
    the denominator but NOT the numerator.
  - Pitfall #7 (post-hoc threshold renegotiation): all locked constants
    are module-top constants asserted at startup; drift halts before any
    live HTTP work.

D-09(P15) carry-forward: audit never persists models to disk. ``models/
xgb_v2.joblib`` is read-only; AUDIT-01 chain checkpoint runs in
Plan 02 Task 4 via ``bash scripts/audit_xgb_v2_sha.sh``.

Usage:
    uv run python scripts/audit_bfo_reachability.py \\
        --output-path .planning/phases/20-audit-spikes-camp-00-bfo-00/BFO_ARCHIVE_REACHABILITY.md

    # Alternate CSV / output paths (rarely needed):
    uv run python scripts/audit_bfo_reachability.py \\
        --targets-csv data/bfo_event_probe_targets.csv \\
        --output-path /tmp/bfo_reachability.md
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup

from ufc_prediction.scraper.bfo_classify import (
    BFO_HOMEPAGE_TITLE,
    EVENT_NOT_FOUND_SENTINEL,
    NON_REACHABLE_STATUSES,
    REACHABLE as REACHABLE_STATUS,
    classify,
    extract_title,
)
from ufc_prediction.scraper.bfo_scraper import BFOParseError, _check_captcha


# ── Locked constants (D-07, D-08) ────────────────────────────────────────

YEAR_ANCHORS: tuple[int, ...] = (2007, 2010, 2013, 2016, 2019, 2022)  # D-07
EVENTS_PER_YEAR: int = 5  # D-07
EARLY_STOP_CONSECUTIVE_FAILURES: int = 5  # D-07
TOTAL_PROBE_BUDGET: int = 30  # D-07 (6×5)
RANDOM_STATE: int = 42  # v2.x

BFO_TIMEOUT_SECONDS: float = 5.0  # D-08
BFO_DELAY_SECONDS: float = 1.2  # D-08
BFO_MAX_RETRIES: int = 3  # D-08

# Phase 21 BFO-V22-03 RED test consumes ``target_rate``; the floor
# guarantees the audit pre-registers an actionable target even when
# measured reachability is below 80%. Phase 21 then has to clear AT LEAST
# this floor regardless of the audit's measured rate.
POST_2007_TARGET_FLOOR: float = 0.80

# Risk-flag thresholds (D-10).
CAPTCHA_RISK_THRESHOLD: float = 0.05  # captcha_rate per year
REACHABILITY_RISK_THRESHOLD: float = 0.60  # reachability_rate per year

TARGETS_CSV: Path = Path("data/bfo_event_probe_targets.csv")
OUTPUT_PATH_DEFAULT: Path = Path(
    ".planning/phases/20-audit-spikes-camp-00-bfo-00/BFO_ARCHIVE_REACHABILITY.md"
)

# EVENT_NOT_FOUND_SENTINEL, BFO_HOMEPAGE_TITLE, REACHABLE_STATUS, and
# NON_REACHABLE_STATUSES are re-exported from bfo_classify (Phase 21
# Plan 21-01 Task 1: single-source-of-truth for the classifier).

logger = logging.getLogger(__name__)


# ── Helpers (underscore-prefixed but exported for tests) ─────────────────


def _is_homepage_fallback_title(html: str) -> bool:
    """Return True if the page <title> matches the BFO generic homepage.

    Phase 20 Task 1 finding kept as a thin convenience wrapper around
    ``bfo_classify.extract_title`` for tests that already imported this
    name. New code should call ``bfo_classify.is_real_event_title``.
    """
    title = extract_title(html)
    return title == BFO_HOMEPAGE_TITLE


def _probe(
    client: Any,
    url: str,
    *,
    anchor_year: int | None = None,
    event_label: str | None = None,
) -> dict[str, Any]:
    """Single-URL probe per Task 1-refined status taxonomy.

    Returns a row dict with at minimum:
      - url, anchor_year, event_label
      - status (one of REACHABLE_STATUS / NON_REACHABLE_STATUSES)
      - elapsed_s (float, monotonic wall-clock seconds)
      - captcha_hit (bool, True only when status == "captcha")
      - http_status_code (int or None)
      - title_extracted (str or None, only on HTTP 200)
      - exc (str or None, exception class+message on error paths)

    Sentinel handling: when ``url == EVENT_NOT_FOUND_SENTINEL`` no HTTP
    request is issued; the row is recorded as ``status="event_not_found"``.
    This counts toward per-year totals but NOT toward reachability.

    Captcha handling (Pitfall #4 / T-20-08): every HTTP 200 is parsed
    with BeautifulSoup and checked via ``_check_captcha(soup)`` from
    ``bfo_scraper.py:191``. On ``BFOParseError`` the row is recorded as
    ``status="captcha"`` with ``captcha_hit=True``. Captcha responses
    are NOT counted as reachable.

    Anti-pattern #3 reminder: DO NOT invoke the BFO scrape orchestrator
    method — that orchestrator's ``parse_errors`` counter will trip on
    event-page HTML structure (event pages have a different schema than
    fighter pages). The audit uses raw ``client.get(url)`` so the
    fighter-page parser never runs.
    """
    base_row: dict[str, Any] = {
        "url": url,
        "anchor_year": anchor_year,
        "event_label": event_label,
        "captcha_hit": False,
        "http_status_code": None,
        "title_extracted": None,
        "exc": None,
    }

    # ── Operator-marked unfindable rows: no HTTP, just sentinel status.
    if url == EVENT_NOT_FOUND_SENTINEL:
        base_row.update(
            {
                "status": "event_not_found",
                "elapsed_s": 0.0,
            }
        )
        return base_row

    t0 = time.monotonic()
    try:
        html = client.get(url)
    except httpx.HTTPStatusError as exc:
        elapsed = time.monotonic() - t0
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code == 404:
            classification = "not_found"
        else:
            classification = "http_other"
        base_row.update(
            {
                "status": classification,
                "elapsed_s": elapsed,
                "http_status_code": status_code,
                "exc": f"{type(exc).__name__}: {exc}",
            }
        )
        return base_row
    except (httpx.TimeoutException,) as exc:
        elapsed = time.monotonic() - t0
        base_row.update(
            {
                "status": "timeout",
                "elapsed_s": elapsed,
                "exc": f"{type(exc).__name__}: {exc}",
            }
        )
        return base_row
    except RuntimeError as exc:
        elapsed = time.monotonic() - t0
        base_row.update(
            {
                "status": "error",
                "elapsed_s": elapsed,
                "exc": f"{type(exc).__name__}: {exc}",
            }
        )
        return base_row

    elapsed = time.monotonic() - t0
    base_row["http_status_code"] = 200
    base_row["elapsed_s"] = elapsed

    # HTTP 200 — check captcha gate FIRST (Pitfall #4 / T-20-08).
    soup = BeautifulSoup(html, "lxml")
    try:
        _check_captcha(soup)
    except BFOParseError as exc:
        base_row.update(
            {
                "status": "captcha",
                "captcha_hit": True,
                "exc": f"{type(exc).__name__}: {exc}",
            }
        )
        return base_row

    # Title-based discrimination (Task 1 finding).
    title_el = soup.find("title")
    title_text = title_el.get_text(strip=True) if title_el is not None else None
    base_row["title_extracted"] = title_text

    if title_text == BFO_HOMEPAGE_TITLE:
        base_row["status"] = "homepage_fallback"
        return base_row

    # Real event page — title is event-specific (not the homepage).
    base_row["status"] = REACHABLE_STATUS
    return base_row


def _probe_year(
    client: Any,
    *,
    year: int,
    candidates: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Probe candidates for one anchor year with early-stop.

    ``candidates`` is a list of ``(event_url, event_label)`` tuples. The
    inner loop tracks ``consecutive_non_reachable`` (any status !=
    REACHABLE_STATUS increments; REACHABLE_STATUS resets to 0). On
    reaching ``EARLY_STOP_CONSECUTIVE_FAILURES`` the loop breaks and
    remaining candidates are NOT probed.

    Sentinel rows (event_url == EVENT_NOT_FOUND_SENTINEL) DO count
    toward consecutive_non_reachable per Task 1 semantics: an anchor
    year where the first 5 rows are all unfindable should still
    trigger early-stop and avoid wasting budget.
    """
    rows: list[dict[str, Any]] = []
    consecutive_non_reachable = 0
    for event_url, event_label in candidates:
        row = _probe(
            client,
            event_url,
            anchor_year=year,
            event_label=event_label,
        )
        rows.append(row)
        if row["status"] == REACHABLE_STATUS:
            consecutive_non_reachable = 0
        else:
            consecutive_non_reachable += 1
            if consecutive_non_reachable >= EARLY_STOP_CONSECUTIVE_FAILURES:
                logger.warning(
                    "[audit-bfo] year=%d early-stop after %d consecutive "
                    "non-reachable rows; %d candidates skipped.",
                    year,
                    consecutive_non_reachable,
                    len(candidates) - len(rows),
                )
                break
    return rows


def _load_event_urls_for_year(
    year: int,
    k: int,
    targets_csv: Path = TARGETS_CSV,
) -> list[tuple[str, str]]:
    """Load (event_url, event_label) tuples for ``year`` from the CSV.

    Raises ``RuntimeError`` if the CSV has fewer than ``k`` rows for the
    given anchor year (fail-fast guard per Open Q #2 Option B).
    """
    with targets_csv.open(newline="", encoding="utf-8") as fh:
        all_rows = [r for r in csv.DictReader(fh) if int(r["anchor_year"]) == year]
    if len(all_rows) < k:
        msg = (
            f"{targets_csv}: expected {k} rows for "
            f"anchor_year={year}, got {len(all_rows)}. Operator must curate."
        )
        raise RuntimeError(msg)
    return [(r["event_url"], r["event_label"]) for r in all_rows[:k]]


def _derive_post2007_target(
    per_year: dict[int, list[dict[str, Any]]],
) -> tuple[float, str]:
    """Mechanically derive the pre-registered post-2007 coverage target.

    Per Task 1 design refinement:
        target_rate = max(POST_2007_TARGET_FLOOR, measured_post_2007_rate)

    where ``measured_post_2007_rate = n_reachable / n_total`` summed over
    all anchor years ≥ 2010 (i.e., the 2007 boundary year is EXCLUDED
    from the denominator — 2007 partial-coverage anomalies must not
    depress the forward-looking target consumed by Phase 21).

    Returns ``(target_rate, rationale_string)``. The rationale string is
    embedded verbatim into ``BFO_ARCHIVE_REACHABILITY.md`` so Phase 21
    plan inputs can cite both the value and its derivation.
    """
    n_reachable = 0
    n_total = 0
    for year, rows in per_year.items():
        if year < 2010:
            # 2007 is the boundary year; excluded from the post-2007 denom.
            continue
        for row in rows:
            n_total += 1
            if row.get("status") == REACHABLE_STATUS:
                n_reachable += 1

    if n_total == 0:
        measured = 0.0
        pct_str = "0.0% (no post-2007 probes)"
    else:
        measured = n_reachable / n_total
        pct_str = f"{measured:.1%}"

    target_rate = max(POST_2007_TARGET_FLOOR, measured)
    rationale = (
        f"max(POST_2007_TARGET_FLOOR={POST_2007_TARGET_FLOOR:.2f}, "
        f"measured_post_2007_reachability={n_reachable}/{n_total}={pct_str}) "
        f"across anchors 2010-2022 (2007 excluded as boundary year). "
        f"Phase 21 BFO-V22-03 RED test gates backfill on this floor."
    )
    return target_rate, rationale


def _compute_risk_flags(
    per_year: dict[int, list[dict[str, Any]]],
) -> list[str]:
    """Per-year risk flags (D-10 §Risk Flags).

    Flags:
      - ``captcha_hit_rate > CAPTCHA_RISK_THRESHOLD`` per year
      - ``reachability_rate < REACHABILITY_RISK_THRESHOLD`` for any
        post-2007 year (2007 is allowed to be < 60% per archive-boundary
        expectations; subsequent years should not be)
    """
    flags: list[str] = []
    for year, rows in per_year.items():
        if not rows:
            continue
        n_total = len(rows)
        n_reachable = sum(1 for r in rows if r.get("status") == REACHABLE_STATUS)
        n_captcha = sum(1 for r in rows if r.get("captcha_hit"))
        captcha_rate = n_captcha / n_total
        reachability_rate = n_reachable / n_total
        if captcha_rate > CAPTCHA_RISK_THRESHOLD:
            flags.append(
                f"year={year}: captcha_hit_rate={captcha_rate:.2f} "
                f"exceeds {CAPTCHA_RISK_THRESHOLD:.2f} threshold "
                f"({n_captcha}/{n_total})"
            )
        if year >= 2010 and reachability_rate < REACHABILITY_RISK_THRESHOLD:
            flags.append(
                f"year={year}: reachability_rate={reachability_rate:.2f} "
                f"below {REACHABILITY_RISK_THRESHOLD:.2f} threshold "
                f"({n_reachable}/{n_total})"
            )
    return flags


def _format_per_year_table(
    per_year: dict[int, list[dict[str, Any]]],
) -> str:
    """Markdown table per D-10 §Per-Year Reachability Table.

    Columns: year | total | reachable | homepage_fallback | not_found |
             http_other | timeout | error | captcha | event_not_found
    """
    header = (
        "| year | total | reachable | homepage_fallback | not_found "
        "| http_other | timeout | error | captcha | event_not_found |"
    )
    sep = (
        "|------|-------|-----------|-------------------|-----------"
        "|------------|---------|-------|---------|-----------------|"
    )
    lines = [header, sep]
    for year in sorted(per_year.keys()):
        rows = per_year[year]
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        n_total = len(rows)
        lines.append(
            f"| {year} | {n_total} "
            f"| {counts.get(REACHABLE_STATUS, 0)} "
            f"| {counts.get('homepage_fallback', 0)} "
            f"| {counts.get('not_found', 0)} "
            f"| {counts.get('http_other', 0)} "
            f"| {counts.get('timeout', 0)} "
            f"| {counts.get('error', 0)} "
            f"| {counts.get('captcha', 0)} "
            f"| {counts.get('event_not_found', 0)} |"
        )
    return "\n".join(lines)


def _format_url_format_drift(
    per_year: dict[int, list[dict[str, Any]]],
) -> str:
    """Markdown section §URL-Format-Drift Findings.

    Reports the canonical URL pattern observed (operator-curated) and
    notes any per-year peculiarities surfaced by the probe (e.g., a
    year where all rows are event_not_found likely indicates the
    pre-archive boundary for that segment).
    """
    lines = [
        "Canonical URL pattern (operator-curated):",
        "`https://www.bestfightodds.com/events/{slug}-{numeric_id}` where",
        "slug is lowercase-hyphenated and numeric_id is 1-5 digits.",
        "",
        "Per-year observations:",
    ]
    for year in sorted(per_year.keys()):
        rows = per_year[year]
        n_total = len(rows)
        n_event_not_found = sum(1 for r in rows if r.get("status") == "event_not_found")
        n_homepage = sum(1 for r in rows if r.get("status") == "homepage_fallback")
        if n_event_not_found == n_total and n_total > 0:
            note = (
                f"  - {year}: all {n_total} rows EVENT_NOT_FOUND — "
                f"likely pre-archive boundary; operator could not locate "
                f"BFO entries via either /events/ archive page or search."
            )
        elif n_event_not_found > 0:
            note = (
                f"  - {year}: {n_event_not_found}/{n_total} EVENT_NOT_FOUND "
                f"(operator-flagged unfindable); "
                f"{n_homepage}/{n_total} homepage_fallback on probe."
            )
        elif n_homepage > 0:
            note = (
                f"  - {year}: {n_homepage}/{n_total} homepage_fallback — "
                f"URL ID likely incorrect or event slug renamed in BFO "
                f"after curation."
            )
        else:
            note = f"  - {year}: no URL-format anomalies detected."
        lines.append(note)
    return "\n".join(lines)


def _format_couture_liddell_note() -> str:
    """Markdown section §Couture–Liddell Substitution Note (D-09).

    Documents why the pre-2007 Couture–Liddell trilogy CANNOT be used
    as the cross-2007-boundary canonical fixture, and substitutes
    Silva–Franklin I (UFC 64, 2006-10-14, pre-archive) → II (UFC 77,
    2007-10-20, in-archive).
    """
    return (
        "The Couture–Liddell trilogy (UFC 43 / 52 / 57; 2003-06-06, "
        "2005-04-16, 2006-02-04) is structurally unreachable on BFO: "
        "the BFO archive starts in 2007 per STACK.md WebFetch "
        "confirmation; pre-2007 events have no BFO event page. "
        "Liddell's last UFC fight was 2010 with no Couture rematch, so "
        "no 2007+ Couture–Liddell match exists either.\n\n"
        "Substitute canonical 2007-boundary fixture per D-09:\n"
        "  - **Anderson Silva vs Rich Franklin I** — UFC 64, "
        "2006-10-14 (pre-archive; not on BFO)\n"
        "  - **Anderson Silva vs Rich Franklin II** — UFC 77, "
        "2007-10-20 (in-archive; "
        "`/events/ufc-77-rich-franklin-vs-anderson-silva-2-237`)\n\n"
        "This pair brackets the 2007 archive boundary with a known-"
        "famous rematch, exercising the same coverage-vs-availability "
        "question as Couture–Liddell would have."
    )


def _write_reachability_md(
    output_path: Path,
    *,
    per_year: dict[int, list[dict[str, Any]]],
    target_rate: float,
    target_rationale: str,
    risk_flags: list[str],
) -> None:
    """Emit BFO_ARCHIVE_REACHABILITY.md with all 6 D-10 sections.

    Section order (D-10, MUST match exactly):
      1. Probe Methodology
      2. Per-Year Reachability Table
      3. URL-Format-Drift Findings
      4. Pre-Registered Post-2007 Coverage Target
      5. Couture–Liddell Substitution Note
      6. Risk Flags
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    n_total = sum(len(rows) for rows in per_year.values())
    n_reachable = sum(
        sum(1 for r in rows if r.get("status") == REACHABLE_STATUS) for rows in per_year.values()
    )
    n_captcha = sum(sum(1 for r in rows if r.get("captcha_hit")) for rows in per_year.values())

    lines: list[str] = [
        "# BFO Archive Reachability — Phase 20 / BFO-V22-00",
        "",
        f"_Generated: {timestamp}_",
        "",
        f"Overall: **{n_reachable}/{n_total} reachable** "
        f"({n_reachable / n_total:.1%} if n_total>0; "
        f"captcha_hits={n_captcha})."
        if n_total > 0
        else f"Overall: 0/0 reachable (no probes recorded).",
        "",
        "## Probe Methodology",
        "",
        f"- Year anchors: `{YEAR_ANCHORS}` (6 anchors)",
        f"- Events per year: `{EVENTS_PER_YEAR}` "
        f"(total budget: `{TOTAL_PROBE_BUDGET}` HTTP requests)",
        f"- Early-stop: `{EARLY_STOP_CONSECUTIVE_FAILURES}` consecutive "
        f"non-reachable rows per year",
        f"- HTTP timeout: `{BFO_TIMEOUT_SECONDS}s`",
        f"- Inter-request delay: `{BFO_DELAY_SECONDS}s` (ScraperClient per-thread rate limit)",
        f"- Max retries: `{BFO_MAX_RETRIES}` "
        f"(ScraperClient exponential backoff 5/10/20s on 429/503/TransportError)",
        f"- Random seed: `{RANDOM_STATE}` (v2.x convention)",
        f"- Targets CSV: `{TARGETS_CSV}` (operator-curated; Option B per RESEARCH §Pitfall #3)",
        "",
        "Status taxonomy (per-row):",
        "- `reachable` — HTTP 200 + event-specific `<title>` (not BFO homepage)",
        "- `homepage_fallback` — HTTP 200 + generic BFO homepage `<title>` (URL ID invalid)",
        "- `not_found` — HTTP 404",
        "- `http_other` — non-200, non-404 HTTP status",
        "- `timeout` — read timeout / connection timeout",
        "- `error` — transport_fail / RuntimeError (retries exhausted)",
        '- `captcha` — HTTP 200 + Cloudflare challenge body (`id="hfmr8"`)',
        "- `event_not_found` — CSV sentinel; operator marked URL as "
        "unfindable (no HTTP request issued)",
        "",
        "## Per-Year Reachability Table",
        "",
        _format_per_year_table(per_year),
        "",
        "## URL-Format-Drift Findings",
        "",
        _format_url_format_drift(per_year),
        "",
        "## Pre-Registered Post-2007 Coverage Target",
        "",
        f"`target_rate >= {target_rate:.2f}`  (rationale: {target_rationale})",
        "",
        "Phase 21 BFO-V22-03 Wave-0 RED test will grep this line exactly; "
        "do not reformat without updating the consumer test.",
        "",
        "## Couture–Liddell Substitution Note",
        "",
        _format_couture_liddell_note(),
        "",
        "## Risk Flags",
        "",
    ]
    if risk_flags:
        for flag in risk_flags:
            lines.append(f"- {flag}")
    else:
        lines.append("None — all per-year reachability and captcha thresholds cleared.")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _assert_locked_constants() -> int:
    """AF startup assertions (D-13 carry-forward): halt on constant drift.

    Returns 0 on success, 2 on any drift detected. Mirrors the
    ``scripts/audit_camp_v22.py`` AF pattern verbatim (stdout/stderr
    shape, non-zero return on mismatch, no live-HTTP work before this
    check passes).
    """
    expected: tuple[tuple[str, Any, Any], ...] = (
        ("YEAR_ANCHORS", YEAR_ANCHORS, (2007, 2010, 2013, 2016, 2019, 2022)),
        ("EVENTS_PER_YEAR", EVENTS_PER_YEAR, 5),
        ("EARLY_STOP_CONSECUTIVE_FAILURES", EARLY_STOP_CONSECUTIVE_FAILURES, 5),
        ("TOTAL_PROBE_BUDGET", TOTAL_PROBE_BUDGET, 30),
        ("RANDOM_STATE", RANDOM_STATE, 42),
        ("BFO_TIMEOUT_SECONDS", BFO_TIMEOUT_SECONDS, 5.0),
        ("BFO_DELAY_SECONDS", BFO_DELAY_SECONDS, 1.2),
        ("BFO_MAX_RETRIES", BFO_MAX_RETRIES, 3),
        ("POST_2007_TARGET_FLOOR", POST_2007_TARGET_FLOOR, 0.80),
    )
    for name, got, want in expected:
        if got != want:
            print(
                f"[audit-bfo] FATAL: {name} drift: expected {want!r}, got {got!r}",
                file=sys.stderr,
            )
            return 2
    print(
        f"[audit-bfo] AF OK: {len(YEAR_ANCHORS)} anchors × {EVENTS_PER_YEAR} "
        f"events = {TOTAL_PROBE_BUDGET} probe budget, "
        f"early-stop={EARLY_STOP_CONSECUTIVE_FAILURES}."
    )
    return 0


# ── argparse + main() ────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 20 / BFO-V22-00 — BFO archive reachability probe "
            "(30 requests across 6 year-anchors)."
        ),
    )
    parser.add_argument(
        "--output-path",
        default=str(OUTPUT_PATH_DEFAULT),
        help="Output path for BFO_ARCHIVE_REACHABILITY.md (default: per D-10).",
    )
    parser.add_argument(
        "--targets-csv",
        default=str(TARGETS_CSV),
        help="Operator-curated CSV of probe URLs (default: per A5).",
    )
    args = parser.parse_args()

    output_path = Path(args.output_path)
    targets_csv = Path(args.targets_csv)
    logging.basicConfig(level=logging.INFO, format="[audit-bfo] %(message)s")

    spike_started = datetime.now(timezone.utc)

    # ── AF startup assertions (Pitfall #7 + D-13 carry-forward) ────────
    af_rc = _assert_locked_constants()
    if af_rc != 0:
        return af_rc

    # ── Fail-fast CSV guard (Pitfall #3 / Open Q #2 Option B) ──────────
    if not targets_csv.exists():
        print(
            f"[audit-bfo] FATAL: targets CSV does not exist: {targets_csv}",
            file=sys.stderr,
        )
        return 2
    with targets_csv.open(newline="", encoding="utf-8") as fh:
        all_rows = list(csv.DictReader(fh))
    if len(all_rows) < TOTAL_PROBE_BUDGET:
        print(
            f"[audit-bfo] FATAL: {targets_csv} has {len(all_rows)} rows, "
            f"need ≥ {TOTAL_PROBE_BUDGET}. Operator must curate.",
            file=sys.stderr,
        )
        return 2

    # ── Lazy import: ScraperClient (keeps test cost low; aligned with audit_camp_v22)
    from ufc_prediction.scraper.client import ScraperClient  # noqa: PLC0415

    client = ScraperClient(
        delay=BFO_DELAY_SECONDS,
        max_retries=BFO_MAX_RETRIES,
        timeout=BFO_TIMEOUT_SECONDS,
    )

    # ── Probe loop (D-07 budget; early-stop per year) ──────────────────
    per_year: dict[int, list[dict[str, Any]]] = {}
    for year in YEAR_ANCHORS:
        print(f"[audit-bfo] year={year}: loading {EVENTS_PER_YEAR} candidates...")
        candidates = _load_event_urls_for_year(
            year,
            k=EVENTS_PER_YEAR,
            targets_csv=targets_csv,
        )
        print(f"[audit-bfo] year={year}: probing {len(candidates)} URLs...")
        rows = _probe_year(client, year=year, candidates=candidates)
        per_year[year] = rows
        n_reachable = sum(1 for r in rows if r.get("status") == REACHABLE_STATUS)
        print(f"[audit-bfo] year={year}: {n_reachable}/{len(rows)} reachable.")

    # ── Derive pre-registered target + risk flags ──────────────────────
    target_rate, target_rationale = _derive_post2007_target(per_year)
    risk_flags = _compute_risk_flags(per_year)

    # ── Emit MD artifact ───────────────────────────────────────────────
    _write_reachability_md(
        output_path,
        per_year=per_year,
        target_rate=target_rate,
        target_rationale=target_rationale,
        risk_flags=risk_flags,
    )
    spike_finished = datetime.now(timezone.utc)

    # ── Final stdout summary (5+ lines per CONTEXT Claude's Discretion) ─
    n_total = sum(len(rows) for rows in per_year.values())
    n_reachable_all = sum(
        sum(1 for r in rows if r.get("status") == REACHABLE_STATUS) for rows in per_year.values()
    )
    n_captcha_all = sum(sum(1 for r in rows if r.get("captcha_hit")) for rows in per_year.values())
    print(
        f"\n[audit-bfo] Audit complete. Wrote:\n"
        f"  {output_path}\n"
        f"[audit-bfo] total_probes           = {n_total}\n"
        f"[audit-bfo] reachable_count        = {n_reachable_all}\n"
        f"[audit-bfo] captcha_hit_count      = {n_captcha_all}\n"
        f"[audit-bfo] target_rate            = {target_rate:.2f}\n"
        f"[audit-bfo] duration               = "
        f"{(spike_finished - spike_started).total_seconds():.0f}s"
    )
    if risk_flags:
        print(f"[audit-bfo] risk_flags = {len(risk_flags)} flag(s) raised")
        for flag in risk_flags:
            print(f"  - {flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
