"""Phase 49 HYG-V26-03 — fighters_names.csv refresh driver.

Refreshes ``data/bfo/fighters_names.csv`` (BFO ↔ DB linkage CSV) using the
canonical matcher path established in Phase 13 + Phase 40-03 architectural
audit: ``bfo_matcher.match_bfo_name`` (rapidfuzz threshold 80) against the
BFO ``/search?query=<name>`` HTML response.

Design lineage: see ``.planning/phases/49-scraper-hygiene-...-/49-FIGHTERS-NAMES-REFRESH-DESIGN.md``.

Closes the Phase 40 Plan 40-03 architectural domain-mismatch carryover:
the Phase 28-04 6-tier matcher targets the ``fighter_aliases`` TABLE
(kaggle ↔ ufcstats), NOT the BFO ↔ DB linkage CSV. This script uses the
right matcher.

Modes:

  --dry-run    Run against a fixture (no DB, no live BFO HTTP). Emits
               the merge log to stdout. Does NOT write the CSV. Useful
               in CI + autonomous review without Supabase credentials.

  --apply      Run against the live DB + live BFO. Reads the current
               CSV baseline, computes the delta (fighters in DB but not
               in CSV), invokes the matcher per delta fighter, emits an
               additive merge to the CSV and a conflicts CSV.

Invariants (enforced):
  - Additive merge ONLY — existing baseline rows in fighters_names.csv
    stay byte-identical (Phase 28-04 operator-reviewed 399 rows)
  - Same-name / different-numeric-id pairs route to
    ``data/bfo/fighters_names_conflicts.csv`` for operator review
    BEFORE the additive merge lands (conflict gate — Phase 40-03 contract)
  - rapidfuzz threshold pinned at MIN_FUZZ_SCORE = 80
  - AUDIT-01: xgb_v2 + meta_v2 SHAs UNCHANGED (data-only refresh)

Backlogged execution: the ``--apply`` mode requires Supabase access and
live BFO connectivity. v2.7+ data-hygiene bucket. Run from a session
with credentials + network egress.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

from ufc_prediction.scraper.bfo_matcher import normalize_name
from ufc_prediction.scraper.bfo_scraper import (
    BFO_BASE,
    EVENT_SLUG_ID_PATTERN,
    MIN_FUZZ_SCORE,
    SEARCH_URL,
    find_bfo_fighter_url,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH: Path = PROJECT_ROOT / "data" / "bfo" / "fighters_names.csv"
DEFAULT_CONFLICTS_PATH: Path = (
    PROJECT_ROOT / "data" / "bfo" / "fighters_names_conflicts.csv"
)
DEFAULT_MERGE_LOG_PATH: Path = (
    PROJECT_ROOT
    / ".planning"
    / "phases"
    / "49-scraper-hygiene-scrape-event-urls-fighters-names-refresh"
    / "49-ALIAS-MERGE-LOG.txt"
)

# BFO fighter href grammar — same shape as EVENT_SLUG_ID_PATTERN but on
# the /fighters/ namespace + CapitalCase slug (BFO renders fighter slugs
# as e.g. "Jon-Jones-819" — uppercase first letter of each word).
# Forbids trailing-dash slugs (the WR-01 / CORPUS-V25-02 bug class)
# inherited from the slug grammar lock in `bfo_scraper.EVENT_SLUG_ID_PATTERN`.
_FIGHTER_HREF_RE = re.compile(
    r"^/fighters/[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?-(\d+)$"
)

# Pin the Phase 28-04 operator-reviewed baseline row count. If the
# baseline CSV grows below this, the additive-merge invariant has been
# violated upstream — halt loudly per AUDIT-01 invariant.
PHASE_28_04_BASELINE_MIN_ROWS = 399


@dataclass(frozen=True)
class DBFighter:
    """Slim DB-fighter view for the refresh driver."""

    id: int
    name: str
    source: str


@dataclass(frozen=True)
class CSVRow:
    """One ``fighters_names.csv`` row."""

    fighter_id: str  # BFO-native (numeric string)
    database: str  # source key (e.g. 'ufcstats')
    name: str  # display name in that source
    database_id: str  # ID in that source (may be empty string)


def _read_csv_baseline(path: Path) -> list[CSVRow]:
    """Read the existing baseline. Halts if row count < Phase 28-04 baseline."""
    if not path.exists():
        logger.warning("fighters_names.csv missing at %s; starting from empty", path)
        return []
    rows: list[CSVRow] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(
                CSVRow(
                    fighter_id=r["fighter_id"],
                    database=r["database"],
                    name=r["name"],
                    database_id=r.get("database_id", "") or "",
                )
            )
    if len(rows) < PHASE_28_04_BASELINE_MIN_ROWS:
        raise RuntimeError(
            f"fighters_names.csv baseline row count {len(rows)} < "
            f"Phase 28-04 baseline {PHASE_28_04_BASELINE_MIN_ROWS}; "
            "additive-merge invariant violated. Halt."
        )
    return rows


def _delta_fighters(
    db_fighters: Iterable[DBFighter],
    baseline_rows: Iterable[CSVRow],
) -> list[DBFighter]:
    """Return DB fighters whose (database, database_id) pair is NOT in the CSV.

    The CSV's ``(database, database_id)`` pair is the BFO↔DB linkage key.
    Any DB fighter whose ``(source, str(id))`` pair is missing from the
    CSV is a candidate for re-matching.
    """
    seen = {(r.database, r.database_id) for r in baseline_rows}
    return [
        f for f in db_fighters
        if (f.source, str(f.id)) not in seen
    ]


def _fetch_search_html(fighter_name: str, http_client) -> Optional[str]:
    """Call BFO ``/search?query=<normalized-name>`` and return HTML body.

    The caller owns rate-limiting + captcha backoff; this is a thin shim
    that defers HTTP semantics to the injected client (same contract as
    ``BFOScraper.scrape_all``).
    """
    url = SEARCH_URL.format(query=normalize_name(fighter_name))
    try:
        return http_client.get(url)
    except Exception as exc:  # noqa: BLE001 - defensive boundary
        logger.warning("BFO search fetch failed for %r: %s", fighter_name, exc)
        return None


def _extract_bfo_numeric_id(profile_url: str) -> Optional[str]:
    """Extract the numeric BFO id from ``https://.../fighters/<Slug>-<N>``.

    Mirrors the slug-id grammar of ``EVENT_SLUG_ID_PATTERN`` for events,
    adapted for the ``/fighters/`` namespace.
    """
    if not profile_url.startswith(BFO_BASE):
        return None
    path = profile_url[len(BFO_BASE):]
    m = _FIGHTER_HREF_RE.match(path)
    return m.group(1) if m is not None else None


@dataclass(frozen=True)
class MatchResult:
    """One match attempt outcome."""

    db_fighter: DBFighter
    matched_url: Optional[str]
    bfo_numeric_id: Optional[str]

    @property
    def matched(self) -> bool:
        return self.matched_url is not None and self.bfo_numeric_id is not None


def _match_one(db_fighter: DBFighter, http_client) -> MatchResult:
    """Run search + match for a single DB fighter."""
    html = _fetch_search_html(db_fighter.name, http_client)
    if html is None:
        return MatchResult(db_fighter=db_fighter, matched_url=None, bfo_numeric_id=None)
    matched_url = find_bfo_fighter_url(
        db_name=db_fighter.name,
        search_html=html,
        threshold=MIN_FUZZ_SCORE,
    )
    bfo_id = _extract_bfo_numeric_id(matched_url) if matched_url else None
    return MatchResult(
        db_fighter=db_fighter,
        matched_url=matched_url,
        bfo_numeric_id=bfo_id,
    )


def _classify(
    match_results: Iterable[MatchResult],
    baseline_rows: Iterable[CSVRow],
) -> tuple[list[CSVRow], list[CSVRow]]:
    """Split match results into (additive_new_rows, conflict_rows).

    Conflict definition: same ``(database, name)`` pair already exists
    in the baseline with a DIFFERENT ``fighter_id`` than the newly
    matched BFO numeric id. Operator review required.
    """
    by_db_name: dict[tuple[str, str], CSVRow] = {
        (r.database, r.name.lower()): r for r in baseline_rows
    }
    new_rows: list[CSVRow] = []
    conflicts: list[CSVRow] = []
    for mr in match_results:
        if not mr.matched:
            continue
        new_row = CSVRow(
            fighter_id=str(mr.bfo_numeric_id),
            database=mr.db_fighter.source,
            name=mr.db_fighter.name,
            database_id=str(mr.db_fighter.id),
        )
        key = (new_row.database, new_row.name.lower())
        existing = by_db_name.get(key)
        if existing is not None and existing.fighter_id != new_row.fighter_id:
            conflicts.append(new_row)
            continue
        if existing is None:
            new_rows.append(new_row)
        # If existing.fighter_id == new_row.fighter_id, it's a re-confirm — skip.
    return (new_rows, conflicts)


def _write_csv_additive(
    path: Path,
    baseline_rows: list[CSVRow],
    new_rows: list[CSVRow],
) -> None:
    """Write baseline + new rows. Baseline order preserved byte-identical."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["fighter_id", "database", "name", "database_id"])
        for r in baseline_rows + new_rows:
            writer.writerow([r.fighter_id, r.database, r.name, r.database_id])


def _write_conflicts(
    path: Path,
    conflicts: list[CSVRow],
) -> None:
    """Write conflicts CSV. Header-only when clean. Rows present means BLOCK."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["fighter_id", "database", "name", "database_id"])
        for r in conflicts:
            writer.writerow([r.fighter_id, r.database, r.name, r.database_id])


def _write_merge_log(
    log_path: Path,
    *,
    baseline_count: int,
    delta_count: int,
    matched_count: int,
    new_rows: int,
    conflicts: int,
    post_total: int,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"# Phase 49 HYG-V26-03 — fighters_names.csv Merge Log\n\n"
        f"Baseline (pre-merge): {baseline_count} rows\n"
        f"Delta (DB fighters not in CSV): {delta_count} candidates\n"
        f"Matched (rapidfuzz >= {MIN_FUZZ_SCORE}): {matched_count}\n"
        f"Additive new rows: {new_rows}\n"
        f"Conflicts (BLOCK pending operator review): {conflicts}\n"
        f"Post-merge total: {post_total} rows\n",
        encoding="utf-8",
    )


# ── DB fetch (apply-mode only) ──────────────────────────────────────────


def _load_db_fighters(database_url: str) -> list[DBFighter]:
    """Load DB fighters with source == 'ufcstats' (Phase 14 canonicalization)."""
    # Deferred import — DB binding is only required in --apply mode and
    # carries a heavy import surface (SQLAlchemy + project models).
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    from ufc_prediction.models.fighter import Fighter

    engine = create_engine(database_url)
    with Session(engine) as session:
        rows = session.execute(
            select(Fighter.id, Fighter.name, Fighter.source).where(
                Fighter.source == "ufcstats"
            )
        ).all()
    return [DBFighter(id=r.id, name=r.name, source=r.source) for r in rows]


# ── Dry-run fixture helpers ─────────────────────────────────────────────


def _dry_run_fixture_fighters() -> list[DBFighter]:
    """Synthetic fighter set for --dry-run mode (no DB access)."""
    return [
        DBFighter(id=1, name="Jon Jones", source="ufcstats"),
        DBFighter(id=2, name="Israel Adesanya", source="ufcstats"),
        DBFighter(id=3, name="Some New Fighter", source="ufcstats"),
    ]


class _DryRunHTTPClient:
    """Returns a synthetic search HTML body for the dry-run fixture."""

    @staticmethod
    def get(url: str) -> Optional[str]:
        # Single hit for the known fighter; empty for unknown.
        lc = url.lower()
        if "jon jones" in lc or "jon+jones" in lc or "jon-jones" in lc:
            return (
                '<html><body>'
                '<a href="/fighters/Jon-Jones-819">Jon Jones</a>'
                '</body></html>'
            )
        return "<html><body>No matches found.</body></html>"


# ── Driver ──────────────────────────────────────────────────────────────


def run(
    *,
    dry_run: bool,
    csv_path: Path = DEFAULT_CSV_PATH,
    conflicts_path: Path = DEFAULT_CONFLICTS_PATH,
    merge_log_path: Path = DEFAULT_MERGE_LOG_PATH,
    database_url: Optional[str] = None,
) -> int:
    """Return process exit code (0 clean / 1 conflicts / 2 error)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    baseline = _read_csv_baseline(csv_path)
    logger.info("Baseline rows: %d", len(baseline))
    if dry_run:
        db_fighters = _dry_run_fixture_fighters()
        http_client = _DryRunHTTPClient()
        logger.info("Dry-run: %d synthetic fighters", len(db_fighters))
    else:
        if database_url is None:
            logger.error("--apply requires --database-url or DATABASE_URL env var")
            return 2
        db_fighters = _load_db_fighters(database_url)
        logger.info("DB fighters: %d (Fighter.source == 'ufcstats')", len(db_fighters))
        # Live HTTP client wiring is the caller's responsibility (e.g.
        # `ScraperClient` from `bfo_scraper`). Deferred to keep this
        # script importable without the full scraper toolchain.
        from ufc_prediction.scraper.scraper_client import ScraperClient
        http_client = ScraperClient(delay_seconds=1.5, max_retries=3)

    delta = _delta_fighters(db_fighters, baseline)
    logger.info("Delta candidates: %d", len(delta))
    match_results = [_match_one(f, http_client) for f in delta]
    matched = [m for m in match_results if m.matched]
    new_rows, conflicts = _classify(matched, baseline)
    logger.info(
        "Matched: %d | New rows: %d | Conflicts: %d",
        len(matched), len(new_rows), len(conflicts),
    )
    if dry_run:
        # Emit merge log to stdout; do NOT touch the CSV.
        sys.stdout.write(
            f"DRY-RUN merge summary:\n"
            f"  baseline={len(baseline)} delta={len(delta)} "
            f"matched={len(matched)} new={len(new_rows)} "
            f"conflicts={len(conflicts)}\n"
        )
        for nr in new_rows:
            sys.stdout.write(f"  + NEW {nr}\n")
        for c in conflicts:
            sys.stdout.write(f"  ! CONFLICT {c}\n")
        return 0 if not conflicts else 1
    if conflicts:
        # Conflict gate — BLOCK before touching the additive CSV. Operator
        # must resolve before re-running with --apply.
        _write_conflicts(conflicts_path, conflicts)
        _write_merge_log(
            merge_log_path,
            baseline_count=len(baseline),
            delta_count=len(delta),
            matched_count=len(matched),
            new_rows=0,
            conflicts=len(conflicts),
            post_total=len(baseline),
        )
        logger.error(
            "BLOCK: %d conflicts emitted to %s. Operator review required "
            "before additive merge.", len(conflicts), conflicts_path,
        )
        return 1
    _write_csv_additive(csv_path, baseline, new_rows)
    _write_conflicts(conflicts_path, [])  # header-only — clean run signal
    _write_merge_log(
        merge_log_path,
        baseline_count=len(baseline),
        delta_count=len(delta),
        matched_count=len(matched),
        new_rows=len(new_rows),
        conflicts=0,
        post_total=len(baseline) + len(new_rows),
    )
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dry-run", action="store_true")
    grp.add_argument("--apply", action="store_true")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--csv-path", default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--conflicts-path", default=str(DEFAULT_CONFLICTS_PATH))
    parser.add_argument("--merge-log-path", default=str(DEFAULT_MERGE_LOG_PATH))
    args = parser.parse_args(argv)
    return run(
        dry_run=args.dry_run,
        csv_path=Path(args.csv_path),
        conflicts_path=Path(args.conflicts_path),
        merge_log_path=Path(args.merge_log_path),
        database_url=args.database_url,
    )


if __name__ == "__main__":
    raise SystemExit(main())
