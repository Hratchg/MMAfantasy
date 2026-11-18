"""Incremental corpus refresh + seed-dump regeneration (weekly CI job and local runbook).

Pulls every UFCStats event newer than the DB's latest, then the data those
fights need, then regenerates the shipped seed dump:

  1. ``ufc scrape latest --backend browser``     new events / fights / stats
  2. ``ufc scrape sherdog --since-year`` +        Sherdog URLs, then debutant seeds
     ``ingest_pre_ufc_records_v25.py --only-missing``
  3. ``ufc scrape odds --fighters-since``         BFO fighter-profile odds, scoped
  4. ``ufc scrape venues --confirm``              venue FKs (unmatched cities reported)
  5. ``ufc elo compute`` + ``ufc features compute``
  6. PG16 bridge -> ``data/seed/ufc_corpus_v30.dump`` (recipe in data/seed/PROVENANCE.md)
  7. sha256 sidecar, ``test_db_seed.py`` goldens and PROVENANCE counts rewritten

Steps 2-4 are soft: a failure is recorded as a warning in the summary and the
refresh continues, because a reviewer reads the summary before merging. Steps
1 and 5-7 are hard failures. With no new events the script exits 0 after
step 1 and writes nothing.

It never touches models or the pinned gate baselines: re-run the retrain gate
and re-anchor (see .claude/skills/retrain-gate/SKILL.md) before merging.

Usage:
    uv run python scripts/refresh_corpus.py --summary refresh_summary.md
    # local dev DB inside a container (host has no pg_dump):
    uv run python scripts/refresh_corpus.py --source-container mmafantasy-db-1
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

from sqlalchemy import create_engine, text

DUMP_PATH = Path("data/seed/ufc_corpus_v30.dump")
SIDECAR_PATH = Path("data/seed/ufc_corpus_v30.dump.sha256")
PROVENANCE_PATH = Path("data/seed/PROVENANCE.md")
GOLDENS_PATH = Path("tests/integration/test_db_seed.py")
BRIDGE_IMAGE = "postgres:16-alpine"
BRIDGE_CONTAINER = "ufc-refresh-pg16"

# Order matches the PROVENANCE table and the EXPECTED_ROW_COUNTS goldens.
SEED_TABLES: tuple[str, ...] = (
    "elo_snapshots",
    "round_stats",
    "computed_features",
    "fight_odds",
    "fights",
    "fighters",
    "events",
    "debutant_seed_inputs",
    "fighter_aliases",
    "venues",
    "referees",
    "alembic_version",
    "model_runs",
)


@dataclass
class RefreshReport:
    since: date | None = None
    new_events: list[tuple[date, str, int, int]] = field(default_factory=list)
    counts_before: dict[str, int] = field(default_factory=dict)
    counts_after: dict[str, int] = field(default_factory=dict)
    unseeded_debutants: list[str] = field(default_factory=list)
    unmatched_venues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dump_sha256: str | None = None
    dump_bytes: int | None = None


# ── Pure helpers (unit-tested) ──────────────────────────────────────────────


def libpq_url(sqlalchemy_url: str) -> str:
    """``postgresql+psycopg://…`` -> ``postgresql://…`` for pg_dump."""
    return re.sub(r"^postgresql\+\w+://", "postgresql://", sqlalchemy_url)


def rewrite_goldens(source: str, counts: dict[str, int]) -> str:
    """Replace the integers in ``EXPECTED_ROW_COUNTS`` with ``counts``."""
    for table, n in counts.items():
        pattern = rf'^(\s*"{table}": )[\d_]+(,)$'
        source, hits = re.subn(pattern, rf"\g<1>{n:_}\g<2>", source, flags=re.MULTILINE)
        if hits != 1:
            raise ValueError(f"expected one golden for {table!r}, found {hits}")
    return source


def rewrite_provenance(
    source: str,
    counts: dict[str, int],
    dump_bytes: int,
    dump_sha256: str,
    generated: date,
    latest_event: date,
) -> str:
    """Update the per-table count block, the size/SHA rows and the Generated line."""
    for table, n in counts.items():
        pattern = rf"^ {re.escape(table)}\s+\|\s+\d+$"
        source, hits = re.subn(pattern, f" {table:<20} | {n:>10}", source, flags=re.MULTILINE)
        if hits != 1:
            raise ValueError(f"expected one PROVENANCE row for {table!r}, found {hits}")
    replacements = [
        (
            r"^\| Compressed dump size \|.*\|$",
            f"| Compressed dump size | {dump_bytes:,} bytes ({dump_bytes / 2**20:.2f} MB) |",
        ),
        (r"^\| SHA256 \|.*\|$", f"| SHA256 | `{dump_sha256}` |"),
        (
            r"^\*\*Generated:\*\*.*$",
            f"**Generated:** {generated.isoformat()} (incremental refresh via "
            f"`scripts/refresh_corpus.py`; corpus current to {latest_event.isoformat()}. "
            "Earlier regenerations: see the notes below and `git log`.)",
        ),
    ]
    for pattern, repl in replacements:
        source, hits = re.subn(pattern, repl, source, flags=re.MULTILINE)
        if hits != 1:
            raise ValueError(f"expected one PROVENANCE match for {pattern!r}, found {hits}")
    return source


def render_summary(report: RefreshReport) -> str:
    if not report.new_events:
        return "## Corpus refresh\n\nNo new UFCStats events. Nothing changed.\n"
    lines = [
        "## Corpus refresh",
        "",
        f"{len(report.new_events)} new UFCStats event(s) since {report.since}.",
        "",
        "| Date | Event | Fights | With odds |",
        "|---|---|---:|---:|",
    ]
    for d, name, n_fights, n_odds in report.new_events:
        lines.append(f"| {d} | {name} | {n_fights} | {n_odds} |")
    lines += ["", "**Row counts**", "", "| Table | Before | After |", "|---|---:|---:|"]
    for table in SEED_TABLES:
        before = report.counts_before.get(table, 0)
        after = report.counts_after.get(table, 0)
        if before != after:
            lines.append(f"| `{table}` | {before:,} | {after:,} |")
    if report.dump_sha256:
        lines += [
            "",
            f"Seed dump: `{report.dump_sha256}` ({(report.dump_bytes or 0) / 2**20:.2f} MB).",
        ]
    if report.unseeded_debutants:
        lines += [
            "",
            f"**Debutants without a Sherdog seed ({len(report.unseeded_debutants)})** "
            "— they get the flat 1500 Elo: " + ", ".join(report.unseeded_debutants),
        ]
    if report.unmatched_venues:
        lines += [
            "",
            "**Unmatched venues** — add rows to `data/venues.csv` + `venues`, then "
            "`ufc scrape venues --confirm`: " + "; ".join(report.unmatched_venues),
        ]
    if report.warnings:
        lines += ["", "**Warnings**", ""] + [f"- {w}" for w in report.warnings]
    lines += [
        "",
        "**Before merging** (the job cannot do these):",
        "- [ ] Restore this dump locally (`ufc db seed --force`) and run `/retrain-gate`.",
        "- [ ] Re-anchor `META_V22_BASELINE_BRIER` and `PLAN_29_03_RANDOM_15PCT_BRIER` "
        "and check the v22 Pitfall E drift (list in `.claude/skills/retrain-gate/SKILL.md`).",
        "",
    ]
    return "\n".join(lines)


# ── DB / subprocess plumbing ────────────────────────────────────────────────


def _run(cmd: list[str], *, soft: bool, report: RefreshReport) -> bool:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    started = time.monotonic()
    rc = subprocess.run(cmd).returncode
    print(f"  -> exit {rc} in {time.monotonic() - started:.0f}s", flush=True)
    if rc == 0:
        return True
    if not soft:
        raise SystemExit(f"step failed (exit {rc}): {' '.join(cmd)}")
    report.warnings.append(
        f"`{' '.join(cmd[2:] if cmd[:2] == ['uv', 'run'] else cmd)}` exited {rc}"
    )
    return False


def _ufc(*args: str) -> list[str]:
    return ["uv", "run", "ufc", *args]


def _counts(conn) -> dict[str, int]:
    return {t: conn.execute(text(f"SELECT count(*) FROM {t}")).scalar_one() for t in SEED_TABLES}


def _latest_event(conn) -> date:
    return conn.execute(text("SELECT max(date) FROM events WHERE source = 'ufcstats'")).scalar_one()


def _new_events(conn, since: date) -> list[tuple[date, str, int, int]]:
    rows = conn.execute(
        text(
            """
            SELECT e.date, e.name, count(DISTINCT f.id), count(DISTINCT o.fight_id)
            FROM events e
            JOIN fights f ON f.event_id = e.id
            LEFT JOIN fight_odds o ON o.fight_id = f.id
            WHERE e.source = 'ufcstats' AND e.date > :since
            GROUP BY e.date, e.name
            ORDER BY e.date
            """
        ),
        {"since": since},
    )
    return [(r[0], r[1], r[2], r[3]) for r in rows]


def _unseeded_debutants(conn, since: date) -> list[str]:
    rows = conn.execute(
        text(
            """
            SELECT fr.name
            FROM fighters fr
            JOIN fights f ON fr.id IN (f.fighter_a_id, f.fighter_b_id)
            JOIN events e ON e.id = f.event_id AND e.source = 'ufcstats'
            WHERE fr.id NOT IN (SELECT fighter_id FROM debutant_seed_inputs)
            GROUP BY fr.id, fr.name
            HAVING min(e.date) > :since
            ORDER BY fr.name
            """
        ),
        {"since": since},
    )
    return [r[0] for r in rows]


def _parse_unmatched_venues(path: Path) -> list[str]:
    if not path.exists():
        return []
    return re.findall(r"^\| \d+ \| (.+) \|$", path.read_text(encoding="utf-8"), flags=re.MULTILINE)


def _bridge_dump(database_url: str, source_container: str | None, out: Path) -> None:
    """PG16-format custom dump of the source DB (see data/seed/PROVENANCE.md)."""
    if source_container:
        source = [
            "docker",
            "exec",
            source_container,
            "pg_dump",
            "-U",
            "ufc",
            "-d",
            "ufc_prediction",
        ]
    else:
        source = ["pg_dump", libpq_url(database_url)]
    source += ["--format=plain", "--no-owner", "--no-privileges"]
    subprocess.run(["docker", "rm", "-f", BRIDGE_CONTAINER], capture_output=True)
    subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", BRIDGE_CONTAINER,
         "-e", "POSTGRES_USER=ufc", "-e", "POSTGRES_PASSWORD=ufc",
         "-e", "POSTGRES_DB=ufc_prediction", BRIDGE_IMAGE],
        check=True, capture_output=True,
    )  # fmt: skip
    try:
        for _ in range(60):
            probe = [
                "docker",
                "exec",
                BRIDGE_CONTAINER,
                "pg_isready",
                "-U",
                "ufc",
                "-d",
                "ufc_prediction",
            ]
            if subprocess.run(probe, capture_output=True).returncode == 0:
                break
            time.sleep(1)
        time.sleep(2)
        producer = subprocess.Popen(source, stdout=subprocess.PIPE)
        loader = subprocess.run(
            ["docker", "exec", "-i", BRIDGE_CONTAINER, "psql", "-q", "-U", "ufc", "-d", "ufc_prediction"],
            stdin=producer.stdout, capture_output=True, text=True,
        )  # fmt: skip
        producer.stdout.close()  # type: ignore[union-attr]
        if producer.wait() != 0:
            raise SystemExit("source pg_dump failed")
        # A newer pg_dump emits settings PG16 does not know (e.g. transaction_timeout);
        # psql skips those statements. Anything else is a real restore error.
        errors = [
            line
            for line in loader.stderr.splitlines()
            if "ERROR" in line and "unrecognized configuration parameter" not in line
        ]
        if errors:
            raise SystemExit("PG16 bridge load failed:\n" + "\n".join(errors[:20]))
        with out.open("wb") as fh:
            subprocess.run(
                ["docker", "exec", BRIDGE_CONTAINER, "pg_dump", "--format=custom", "-Z", "9",
                 "--no-owner", "--no-privileges", "-U", "ufc", "-d", "ufc_prediction"],
                stdout=fh, check=True,
            )  # fmt: skip
    finally:
        subprocess.run(["docker", "stop", BRIDGE_CONTAINER], capture_output=True)


def _write_github_output(new_events: int) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"new_events={new_events}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", type=Path, help="Write the markdown summary here")
    parser.add_argument(
        "--source-container",
        help="Run pg_dump inside this container (local dev DB) instead of on the host",
    )
    parser.add_argument("--scrape-backend", default="browser", choices=["browser", "http"])
    args = parser.parse_args(argv)

    from ufc_prediction.config import settings

    database_url = settings.database_url
    engine = create_engine(database_url)
    report = RefreshReport()

    with engine.connect() as conn:
        report.since = _latest_event(conn)
        report.counts_before = _counts(conn)
    print(f"latest ufcstats event before refresh: {report.since}", flush=True)

    _run(_ufc("scrape", "latest", "--backend", args.scrape_backend), soft=False, report=report)

    with engine.connect() as conn:
        report.new_events = _new_events(conn, report.since)
    if not report.new_events:
        print("no new events — nothing to do", flush=True)
        _write_github_output(0)
        if args.summary:
            args.summary.write_text(render_summary(report), encoding="utf-8")
        return 0

    first_new = report.new_events[0][0]
    if _run(
        _ufc("scrape", "sherdog", "--since-year", str(first_new.year)), soft=True, report=report
    ):
        _run(
            ["uv", "run", "python", "scripts/ingest_pre_ufc_records_v25.py", "--only-missing"],
            soft=True,
            report=report,
        )
    _run(
        _ufc(
            "scrape",
            "odds",
            "--min-date",
            first_new.isoformat(),
            "--fighters-since",
            first_new.isoformat(),
        ),
        soft=True,
        report=report,
    )
    unmatched_report = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "refresh-unmatched-venues.md"
    unmatched_report.unlink(missing_ok=True)
    _run(
        _ufc("scrape", "venues", "--confirm", "--unmatched-output", str(unmatched_report)),
        soft=True,
        report=report,
    )
    _run(_ufc("elo", "compute"), soft=False, report=report)
    _run(_ufc("features", "compute"), soft=False, report=report)

    with engine.connect() as conn:
        report.new_events = _new_events(conn, report.since)
        report.counts_after = _counts(conn)
        report.unseeded_debutants = _unseeded_debutants(conn, report.since)
        latest = _latest_event(conn)
    report.unmatched_venues = _parse_unmatched_venues(unmatched_report)

    print("\nregenerating the seed dump through the PG16 bridge…", flush=True)
    _bridge_dump(database_url, args.source_container, DUMP_PATH)
    payload = DUMP_PATH.read_bytes()
    report.dump_bytes = len(payload)
    report.dump_sha256 = hashlib.sha256(payload).hexdigest()
    SIDECAR_PATH.write_text(f"{report.dump_sha256}  {DUMP_PATH.name}\n", encoding="utf-8")
    GOLDENS_PATH.write_text(
        rewrite_goldens(GOLDENS_PATH.read_text(encoding="utf-8"), report.counts_after),
        encoding="utf-8",
    )
    PROVENANCE_PATH.write_text(
        rewrite_provenance(
            PROVENANCE_PATH.read_text(encoding="utf-8"),
            report.counts_after,
            report.dump_bytes,
            report.dump_sha256,
            datetime.now(UTC).date(),
            latest,
        ),
        encoding="utf-8",
    )

    summary = render_summary(report)
    print("\n" + summary, flush=True)
    if args.summary:
        args.summary.write_text(summary, encoding="utf-8")
    _write_github_output(len(report.new_events))
    return 0


if __name__ == "__main__":
    sys.exit(main())
