"""Sherdog pre-UFC record ingestion CLI (DEBUT-V25-01, Plan 43-01 Task 2).

Enumerates every UFC debutant (every fighter, at their first UFC appearance
they have 0 prior UFC fights), scrapes their Sherdog profile via the existing
ScraperClient (>=2.0s polite rate-limit), filters to pre-UFC fights only,
computes a PreUFCRecord, derives last_organization + org_tier, and
UPSERTs into ``data/sherdog/pre_ufc_records.csv`` (14-column locked schema).

Reuse: ``filter_pre_ufc_fights`` + ``compute_pre_ufc_stats`` from
``src/ufc_prediction/scraper/sherdog.py`` (Phase 22-04 substrate).

Anti-bot policy: ``detect_antibot()`` HALTs the run on Cloudflare / 403 /
429 / 503 signatures (Phase 40 trust precedent — no silent retries beyond
the existing ScraperClient backoff).

Failure handling: per-fighter exceptions land in
``data/sherdog/pre_ufc_records_failures.csv`` with a ``reason`` column;
the main run continues.

Idempotency: ``upsert_csv()`` keys on ``fighter_id`` — re-running the
script does NOT duplicate rows (replaces in place).

CLI:
    python scripts/ingest_pre_ufc_records_v25.py \\
        --output data/sherdog/pre_ufc_records.csv \\
        --failures-output data/sherdog/pre_ufc_records_failures.csv \\
        [--limit N] [--start-from-fighter-id N]
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

# Project source on path when invoked as `python scripts/...`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ufc_prediction.scraper.sherdog import (
    compute_pre_ufc_stats,
    filter_pre_ufc_fights,
    parse_sherdog_fighter_page,
)
from ufc_prediction.scraper.sherdog_models import (
    PreUFCRecord,
    SherdogFight,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Constants — locked by Plan 43-01 CONTEXT
# ─────────────────────────────────────────────────────────────────────────

# Substring tokens (case-insensitive). Order: more-specific first to avoid
# false positives. Sherdog event_names embed the org (e.g., "Bellator 290:
# Bader vs Fedor"), so substring containment is correct here.
MAJOR_ORGS: tuple[str, ...] = (
    "Bellator",
    "ONE Championship",
    "ONE FC",
    "PFL",
    "Professional Fighters League",
    "Pride",
    "Strikeforce",
    "WEC",
    "World Extreme Cagefighting",
    "RIZIN",
)

REGIONAL_ORGS: tuple[str, ...] = (
    "LFA",
    "Legacy Fighting Alliance",
    "KSW",
    "Cage Warriors",
    "M-1",
    "Brave CF",
    "Brave Combat Federation",
    "Eagle FC",
    "Combate Global",
    "Combate Americas",
)

# Locked 14-column schema (43-01-PLAN.md must_haves).
CSV_COLUMNS: list[str] = [
    "fighter_id",
    "sherdog_url",
    "n_pre_ufc_fights",
    "wins",
    "losses",
    "draws",
    "nc_dq",
    "win_rate",
    "kos",
    "submissions",
    "decisions",
    "last_organization",
    "org_tier",
    "scraped_at",
]

FAILURES_COLUMNS: list[str] = [
    "fighter_id",
    "sherdog_url",
    "reason",
    "logged_at",
]

OrgTier = Literal["major", "regional", "local", "none"]


# Cloudflare / anti-bot HTML signatures (case-sensitive substring match
# — Cloudflare emits these verbatim).
_ANTIBOT_HTML_SIGNATURES: tuple[str, ...] = (
    "Just a moment...",
    "cf-browser-verification",
    "Cloudflare Ray ID",
    "Attention Required! | Cloudflare",
)

_ANTIBOT_STATUS_CODES: frozenset[int] = frozenset({403, 429, 503})


# ─────────────────────────────────────────────────────────────────────────
# Pure helpers (testable; no DB / no network)
# ─────────────────────────────────────────────────────────────────────────


def classify_org_tier(org_name: str | None) -> OrgTier:
    """Classify an organization name into the locked 4-tier taxonomy.

    Returns:
        "major" if the name contains a major-org token (case-insensitive).
        "regional" if the name contains a regional-org token.
        "local" for any other non-empty named org.
        "none" if the name is None / empty / whitespace-only.
    """
    if org_name is None:
        return "none"
    cleaned = org_name.strip()
    if not cleaned:
        return "none"
    lower = cleaned.lower()
    for token in MAJOR_ORGS:
        if token.lower() in lower:
            return "major"
    for token in REGIONAL_ORGS:
        if token.lower() in lower:
            return "regional"
    return "local"


def derive_last_organization(pre_ufc_fights: list[SherdogFight]) -> str | None:
    """Return the event_name of the most recent dated pre-UFC fight.

    Strips trailing event-number / colon block (e.g., "Bellator 290: Bader
    vs Fedor" -> "Bellator"). Returns None if no fights have an event_date.
    """
    dated = [f for f in pre_ufc_fights if f.event_date is not None]
    if not dated:
        return None
    dated_sorted = sorted(dated, key=lambda f: f.event_date, reverse=True)  # type: ignore[arg-type, return-value]
    name = dated_sorted[0].event_name
    if not name:
        return None
    # Strip a trailing " <number>:..." block to recover the org root.
    # "Bellator 290: Bader vs Fedor" -> "Bellator"
    # "Cage Warriors 50" -> "Cage Warriors" (no trailing colon-block — keep)
    # "ONE Championship: Lights Out" -> "ONE Championship"
    stripped = re.sub(r"\s+\d+\s*:.*$", "", name).strip()
    stripped = re.sub(r":\s*.*$", "", stripped).strip()
    return stripped or name


def build_csv_row(
    fighter_id: int,
    sherdog_url: str | None,
    record: PreUFCRecord,
    last_org: str | None,
    tier: str,
    scraped_at: str,
) -> dict[str, object]:
    """Construct a CSV row matching the locked 14-column schema."""
    wins = record.total_wins
    losses = record.total_losses
    draws = record.total_draws
    nc_dq = record.total_fights - wins - losses - draws
    return {
        "fighter_id": fighter_id,
        "sherdog_url": sherdog_url if sherdog_url is not None else "",
        "n_pre_ufc_fights": record.total_fights,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "nc_dq": nc_dq,
        "win_rate": record.win_pct,
        "kos": round(record.ko_finish_rate * wins),
        "submissions": round(record.sub_finish_rate * wins),
        "decisions": round(record.decision_rate * wins),
        "last_organization": last_org if last_org is not None else "",
        "org_tier": tier,
        "scraped_at": scraped_at,
    }


def upsert_csv(path: Path, rows: list[dict]) -> int:
    """UPSERT rows into a CSV keyed on fighter_id (idempotent).

    Reads any existing CSV into a dict keyed on fighter_id, overlays the
    new rows (conflicts: new wins), writes atomically via tempfile+rename.

    Returns the final row count.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict[str, object]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                fid = str(r.get("fighter_id", "")).strip()
                if fid:
                    existing[fid] = r

    for row in rows:
        fid = str(row["fighter_id"])
        # Re-normalize to the locked column set + stringify (CSV is text).
        existing[fid] = {col: row.get(col, "") for col in CSV_COLUMNS}

    # Sort by fighter_id (numeric where possible) for stable diffs.
    def _sort_key(item: tuple[str, dict]) -> tuple[int, str]:
        fid = item[0]
        try:
            return (0, f"{int(fid):020d}")
        except ValueError:
            return (1, fid)

    sorted_rows = [v for _, v in sorted(existing.items(), key=_sort_key)]

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as fh:
        # lineterminator="\n" — match POSIX convention so plan-verify and
        # downstream tooling can compare headers with $(head -1 ...) without
        # `\r` artifacts. RFC 4180 \r\n is harmless for CSV readers, but bash
        # text comparison treats the \r as part of the field.
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for r in sorted_rows:
            writer.writerow({col: r.get(col, "") for col in CSV_COLUMNS})
    os.replace(tmp_path, path)
    return len(sorted_rows)


def write_failures_csv(path: Path, failures: list[dict]) -> int:
    """Overwrite the failures CSV with the current run's failure list."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=FAILURES_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in failures:
            writer.writerow({col: row.get(col, "") for col in FAILURES_COLUMNS})
    return len(failures)


def detect_antibot(html: str, status_code: int) -> bool:
    """Detect Sherdog anti-bot challenge (Cloudflare / rate-limit gate).

    Returns True on:
    - HTTP status in {403, 429, 503}
    - HTML containing any known Cloudflare challenge signature
    """
    if status_code in _ANTIBOT_STATUS_CODES:
        return True
    if not html:
        return False
    for sig in _ANTIBOT_HTML_SIGNATURES:
        if sig in html:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────
# DB helper
# ─────────────────────────────────────────────────────────────────────────


def enumerate_debutants(session: object) -> list[tuple[int, str | None, date]]:
    """Enumerate every UFC debutant — i.e., every fighter who has fought
    in the UFC at least once, returned with their first UFC fight date.

    Every fighter is a debutant at their first UFC appearance (n_ufc_fights
    == 0 before that fight), so the practical filter is: every fighter
    with at least one UFC fight in the corpus.

    Returns: list of (fighter_id, sherdog_url, first_ufc_date), ordered by
    first_ufc_date ASC for deterministic processing.
    """
    from sqlalchemy import func as sa_func

    from ufc_prediction.models.event import Event
    from ufc_prediction.models.fight import Fight
    from ufc_prediction.models.fighter import Fighter

    # Subquery: per-fighter min(event.date) across UFC fights.
    # (Event.source filter not needed — corpus contract is UFC-only via
    # source='ufcstats'; if non-UFC sources exist in dev DBs they are
    # treated as audit-only and silently excluded by the inner join
    # boundary — see Event model docstring.)
    fighter_id_min_date = (
        session.query(  # type: ignore[attr-defined]
            Fighter.id.label("fighter_id"),
            Fighter.sherdog_url.label("sherdog_url"),
            sa_func.min(Event.date).label("first_ufc_date"),
        )
        .join(
            Fight,
            (Fight.fighter_a_id == Fighter.id) | (Fight.fighter_b_id == Fighter.id),
        )
        .join(Event, Fight.event_id == Event.id)
        .filter(Event.source == "ufcstats")
        .group_by(Fighter.id, Fighter.sherdog_url)
        .order_by(sa_func.min(Event.date).asc())
    )
    out: list[tuple[int, str | None, date]] = []
    for row in fighter_id_min_date.all():
        # SQLAlchemy Row supports both .name access and tuple indexing.
        # Test fixtures pass plain 3-tuples → use index access for both.
        if isinstance(row, tuple):
            out.append((row[0], row[1], row[2]))
        else:
            out.append((row.fighter_id, row.sherdog_url, row.first_ufc_date))
    return out


# ─────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────


def _fetch_with_status(client: object, url: str) -> tuple[str, int]:
    """Wrap ScraperClient.get() to also report an HTTP status code.

    ScraperClient.get() raises on non-200 / non-retryable codes (and surfaces
    raise_for_status() on others), so the success path returns (html, 200).
    Anti-bot 403/429/503 reach us only if they exhaust retries — surfaced
    here as an httpx.HTTPStatusError, which we map to (msg, status_code) so
    detect_antibot() can act.
    """
    import httpx

    try:
        html = client.get(url)  # type: ignore[attr-defined]
        return html, 200
    except httpx.HTTPStatusError as exc:
        body = exc.response.text if exc.response is not None else ""
        code = exc.response.status_code if exc.response is not None else 0
        return body, code


def _write_antibot_halt(
    halt_path: Path,
    url: str,
    status_code: int,
    body_snippet: str,
) -> None:
    halt_path.parent.mkdir(parents=True, exist_ok=True)
    iso = datetime.now(UTC).isoformat()
    halt_path.write_text(
        "Sherdog anti-bot challenge detected — HALTING per Phase 40 trust precedent.\n"
        f"timestamp_utc={iso}\n"
        f"url={url}\n"
        f"status_code={status_code}\n"
        "body_first_500_chars=\n"
        f"{body_snippet[:500]}\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ingest_pre_ufc_records_v25",
        description=(
            "Scrape Sherdog pre-UFC records across all UFC debutants and "
            "UPSERT into data/sherdog/pre_ufc_records.csv (Plan 43-01)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sherdog/pre_ufc_records.csv"),
        help="Output CSV path (default: data/sherdog/pre_ufc_records.csv)",
    )
    parser.add_argument(
        "--failures-output",
        type=Path,
        default=Path("data/sherdog/pre_ufc_records_failures.csv"),
        help="Failures CSV path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N debutants (for testing; default = all)",
    )
    parser.add_argument(
        "--start-from-fighter-id",
        type=int,
        default=None,
        help="Skip every fighter with id < this value (resumption helper)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.5,
        help="Polite rate-limit delay (seconds) between Sherdog requests "
        "(default 2.5 — Phase 22 used 2.0; bump for v2.5 conservatism)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Pre-flight: imports here (lazy) so --help doesn't require DB env.
    from ufc_prediction.db.session import SessionLocal
    from ufc_prediction.scraper.client import ScraperClient

    halt_path = args.output.parent / "pre_ufc_records_ANTIBOT_HALT.txt"
    if halt_path.exists():
        logger.warning(
            "Stale ANTIBOT_HALT file present at %s — removing before run",
            halt_path,
        )
        halt_path.unlink()

    started = datetime.now(UTC)
    started_iso = started.isoformat()
    logger.info("=== Plan 43-01 DEBUT-V25-01 ingestion start: %s ===", started_iso)
    logger.info("output=%s", args.output)
    logger.info("failures_output=%s", args.failures_output)
    logger.info(
        "delay=%.2fs limit=%s start_from_fighter_id=%s",
        args.delay,
        args.limit,
        args.start_from_fighter_id,
    )

    session = SessionLocal()
    try:
        debutants = enumerate_debutants(session)
    finally:
        session.close()
    total_enumerated = len(debutants)
    logger.info("enumerated debutants: %d", total_enumerated)

    if args.start_from_fighter_id is not None:
        debutants = [d for d in debutants if d[0] >= args.start_from_fighter_id]
        logger.info("after --start-from-fighter-id filter: %d", len(debutants))
    if args.limit is not None:
        debutants = debutants[: args.limit]
        logger.info("after --limit: %d", len(debutants))

    client = ScraperClient(delay=args.delay)
    new_rows: list[dict] = []
    failures: list[dict] = []
    ok_count = 0
    failure_reason_counts: dict[str, int] = {}

    def _record_failure(fighter_id: int, sherdog_url: str | None, reason: str) -> None:
        nonlocal failures
        failures.append(
            {
                "fighter_id": fighter_id,
                "sherdog_url": sherdog_url or "",
                "reason": reason,
                "logged_at": datetime.now(UTC).isoformat(),
            }
        )
        failure_reason_counts[reason] = failure_reason_counts.get(reason, 0) + 1

    try:
        for idx, (fighter_id, sherdog_url, first_ufc_date) in enumerate(debutants):
            if (idx % 50) == 0:
                logger.info(
                    "progress: %d/%d (ok=%d failed=%d)",
                    idx,
                    len(debutants),
                    ok_count,
                    len(failures),
                )

            if not sherdog_url:
                _record_failure(fighter_id, None, "no_sherdog_url")
                continue

            try:
                html, status_code = _fetch_with_status(client, sherdog_url)
            except Exception as exc:
                reason = f"fetch_error:{type(exc).__name__}"
                _record_failure(fighter_id, sherdog_url, reason)
                logger.warning(
                    "fetch failed for fighter_id=%d url=%s reason=%s",
                    fighter_id,
                    sherdog_url,
                    reason,
                )
                continue

            if detect_antibot(html, status_code):
                _write_antibot_halt(halt_path, sherdog_url, status_code, html)
                logger.error(
                    "ANTI-BOT HALT — Sherdog returned status=%d / Cloudflare "
                    "signature on url=%s. Wrote %s. Exiting per Phase 40 "
                    "trust precedent.",
                    status_code,
                    sherdog_url,
                    halt_path,
                )
                # Persist what we have so far (atomic UPSERT) before exit.
                upsert_csv(args.output, new_rows)
                write_failures_csv(args.failures_output, failures)
                return 2

            try:
                profile = parse_sherdog_fighter_page(html, sherdog_url)
                pre_ufc = filter_pre_ufc_fights(profile.fights, first_ufc_date)
                record = compute_pre_ufc_stats(pre_ufc)
                last_org = derive_last_organization(pre_ufc)
                tier = classify_org_tier(last_org)
                row = build_csv_row(
                    fighter_id=fighter_id,
                    sherdog_url=sherdog_url,
                    record=record,
                    last_org=last_org,
                    tier=tier,
                    scraped_at=datetime.now(UTC).isoformat(),
                )
                new_rows.append(row)
                ok_count += 1
            except Exception as exc:
                reason = f"parse_error:{type(exc).__name__}"
                _record_failure(fighter_id, sherdog_url, reason)
                logger.warning(
                    "parse failed for fighter_id=%d url=%s reason=%s",
                    fighter_id,
                    sherdog_url,
                    reason,
                )

            # Periodic flush every 100 successful scrapes (resilience).
            if ok_count and (ok_count % 100) == 0 and new_rows:
                upsert_csv(args.output, new_rows)
                new_rows = []
                write_failures_csv(args.failures_output, failures)
                logger.info(
                    "flushed checkpoint: ok=%d failed=%d",
                    ok_count,
                    len(failures),
                )
    finally:
        client.close()

    # Final flush
    final_count = upsert_csv(args.output, new_rows)
    write_failures_csv(args.failures_output, failures)

    elapsed = (datetime.now(UTC) - started).total_seconds()
    logger.info("=== Plan 43-01 DEBUT-V25-01 ingestion done ===")
    logger.info("total_enumerated=%d", total_enumerated)
    logger.info("ok=%d", ok_count)
    logger.info("failed=%d", len(failures))
    for reason, count in sorted(failure_reason_counts.items()):
        logger.info("  failure_reason=%s count=%d", reason, count)
    logger.info("final_csv_row_count=%d", final_count)
    logger.info("elapsed_seconds=%.1f", elapsed)
    logger.info("antibot_triggered=NO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
