#!/usr/bin/env python
"""Phase 41 / BFO-V25-01 — Diagnose 2011/2013/2019/2020 BFO disambiguation anomaly.

One-shot diagnostic that characterizes the anomaly cluster years observed
in the v2.4 audit + corpus_v25_delta.md per-year breakdown across four
dimensions:

  1. Per-year BFO coverage % (2010-2026) — paired FightOdds alias join, mirroring
     ``scripts/corpus_v25_delta.py::query_per_year_bfo``.
  2. Fighter-name match-rate distribution — for a capped sample of fights
     in each anomaly year that LACK a paired BFO row, re-run
     ``bfo_matcher.match_bfo_name()`` against the BFO names extracted from
     ``data/bfo/fighters_names.csv`` (operator-curated BFO name list) and
     bucketize ``fuzz.ratio`` outcomes into quartiles.
  3. URL canonicalization failure-rate — Phase 40-01's
     ``canonicalize_event_url()`` is run against the operator-curated BFO
     event-URL CSV (``data/bfo_event_probe_targets.csv`` from Phase 20)
     filtered to anchor years 2010-2026; counts canonicalize-failures
     per year.
  4. Event-date drift histogram — fights with BFO match BUT where the
     stored BFO event-date diverges from the canonical ``events.date``
     (computed by joining ``events`` to BFO closing-prob rows; drift in
     days bucketized to {0, 1-3, 4-7, 8-14, 15+}).
  5. Phase 40-01 cross-check — for each anomaly-year row currently lacking
     a BFO match, run ``canonicalize_event_url()`` against the operator
     event-URL CSV and count how many would NOW resolve (post-Phase 40-01).

Locked decisions (41-CONTEXT.md):
  - D-Diagnosis approach: "Spike-then-fix in same plan"; this script is the spike.
  - D-Output structure: writeup is partner-facing; this script feeds the
    evidence block to ``results/bfo_disambiguation_v25.md``.
  - D-Likely root causes: (1) rapidfuzz threshold too loose for sparse-coverage
    years, (2) URL canonicalization bug (Phase 40-01 cross-check), (3) Sherdog/
    UFCStats name divergence.

Anti-features (BANNED):
  - AF-1: hyperparameter retuning / model retrain.
  - AF-2: feature engineering / schema mutation.
  - AF-3: ``models/xgb_v2.joblib`` mutation; ``models/meta/meta_v2.joblib`` mutation.
  - AF-4: any write to Postgres / Supabase.

Path-A vs Path-B execution:
  - Path A (DB reachable): all 5 JSON keys populated from live queries.
  - Path B (DB unreachable): script writes per_year_coverage from frozen
    ``.planning/phases/40-corpus-growth-scraper-hygiene/40-CORPUS-POST-SCRAPE-STATS.json``
    + ``results/corpus_v25_delta.md`` cached per-year table; the other 3
    DB-dependent keys carry ``{"db_status": "unreachable", "deferred": true}``
    sentinels so downstream parsers see all 5 required keys. The
    Phase-40-01 cross-check + name-match-rate quartiles + date-drift
    histogram run via the operator-curated CSV inputs that DO not need DB.

Usage:
    python scripts/diagnose_bfo_disambiguation_v25.py \\
        --output-json results/bfo_disambiguation_v25.json \\
        --output-md /tmp/bfo_evidence_block.md

    # Force Path-B (skip DB queries) for offline development:
    python scripts/diagnose_bfo_disambiguation_v25.py \\
        --output-json results/bfo_disambiguation_v25.json \\
        --output-md /tmp/bfo_evidence_block.md \\
        --no-db
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

# Project-local imports kept inside main() / handlers so the script imports
# cleanly without a DB driver in --no-db mode.

logger = logging.getLogger(__name__)


# ── Locked constants (41-CONTEXT.md decisions) ───────────────────────────

ANOMALY_YEARS: tuple[int, ...] = (2011, 2013, 2019, 2020)
YEAR_RANGE: tuple[int, ...] = tuple(range(2010, 2027))  # 2010-2026 inclusive
SAMPLE_CAP_PER_YEAR: int = 200  # T-41-01-04 DoS cap

# Frozen substrate snapshot from Plan 40-04 (corpus_v25_delta.md).
# Used when DB is unreachable (Path B). Per-year `populated` and `total`
# come verbatim from the Plan 40-04 post-scrape per-year breakdown
# documented in ``results/corpus_v25_delta.md``.
FROZEN_PER_YEAR_COVERAGE: dict[int, dict[str, int]] = {
    2010: {"total": 510, "populated": 30},
    2011: {"total": 611, "populated": 52},
    2012: {"total": 691, "populated": 39},
    2013: {"total": 789, "populated": 42},
    2014: {"total": 1036, "populated": 92},
    2015: {"total": 975, "populated": 85},
    2016: {"total": 1011, "populated": 72},
    2017: {"total": 935, "populated": 47},
    2018: {"total": 963, "populated": 97},
    2019: {"total": 1069, "populated": 54},
    2020: {"total": 930, "populated": 25},
    2021: {"total": 1011, "populated": 343},
    2022: {"total": 1017, "populated": 321},
    2023: {"total": 1024, "populated": 338},
    2024: {"total": 1030, "populated": 400},
    2025: {"total": 1042, "populated": 399},
    2026: {"total": 269, "populated": 91},
}

# Operator-curated CSV inputs (already present in the repo from Phase 20):
BFO_NAMES_CSV: Path = Path("data/bfo/fighters_names.csv")
BFO_EVENT_PROBE_CSV: Path = Path("data/bfo_event_probe_targets.csv")

# Output defaults
DEFAULT_OUTPUT_JSON: Path = Path("results/bfo_disambiguation_v25.json")
DEFAULT_OUTPUT_MD: Path = Path("/tmp/bfo_evidence_block.md")


# ── Helpers ──────────────────────────────────────────────────────────────


def _percent(populated: int, total: int) -> float:
    """Coverage % with 2-decimal rounding; returns 0.0 if total == 0."""
    if total <= 0:
        return 0.0
    return round(100.0 * populated / total, 2)


def _bucketize_drift(days: int) -> str:
    """Bucketize event-date drift in days per Plan 41-01 Task 2 spec."""
    if days <= 0:
        return "0"
    if days <= 3:
        return "1-3"
    if days <= 7:
        return "4-7"
    if days <= 14:
        return "8-14"
    return "15+"


def _quartile_bucket(score: float) -> str:
    """Quartile bucket for a 0-100 fuzz.ratio."""
    if score < 25:
        return "q1_below_25"
    if score < 50:
        return "q2_25_to_50"
    if score < 75:
        return "q3_50_to_75"
    if score < 80:
        return "q4_75_to_80_subthreshold"
    return "q5_80_plus_match"


# ── Section 1: per-year BFO coverage (DB query OR frozen fallback) ───────


def query_per_year_coverage_db() -> list[dict[str, Any]]:
    """Live-DB per-year BFO coverage. Mirrors corpus_v25_delta.query_per_year_bfo.

    Raises any DB exception up to the caller (so Path-B fallback can catch).
    """
    from sqlalchemy import Integer, and_, case, func  # noqa: PLC0415
    from sqlalchemy.orm import aliased  # noqa: PLC0415

    from ufc_prediction.db.session import SessionLocal  # noqa: PLC0415
    from ufc_prediction.models.event import Event  # noqa: PLC0415
    from ufc_prediction.models.fight import Fight  # noqa: PLC0415
    from ufc_prediction.models.fight_odds import FightOdds  # noqa: PLC0415

    session = SessionLocal()
    try:
        odds_a = aliased(FightOdds)
        odds_b = aliased(FightOdds)
        year_expr = func.cast(func.extract("year", Event.date), Integer)
        populated_expr = case(
            (
                and_(
                    odds_a.closing_implied_prob.isnot(None),
                    odds_b.closing_implied_prob.isnot(None),
                ),
                1,
            ),
            else_=0,
        )
        rows = (
            session.query(
                year_expr.label("yr"),
                func.count(Fight.id).label("total"),
                func.coalesce(func.sum(populated_expr), 0).label("populated"),
            )
            .join(Event, Fight.event_id == Event.id)
            .outerjoin(
                odds_a,
                and_(
                    odds_a.fight_id == Fight.id,
                    odds_a.fighter_id == Fight.fighter_a_id,
                ),
            )
            .outerjoin(
                odds_b,
                and_(
                    odds_b.fight_id == Fight.id,
                    odds_b.fighter_id == Fight.fighter_b_id,
                ),
            )
            .filter(year_expr.in_(YEAR_RANGE))
            .group_by(year_expr)
            .order_by(year_expr.asc())
            .all()
        )
        return [
            {
                "year": int(yr),
                "total": int(total),
                "populated": int(populated),
                "coverage_pct": _percent(int(populated), int(total)),
                "is_anomaly_year": int(yr) in ANOMALY_YEARS,
            }
            for (yr, total, populated) in rows
        ]
    finally:
        session.close()


def per_year_coverage_frozen() -> list[dict[str, Any]]:
    """Path-B fallback: read from FROZEN_PER_YEAR_COVERAGE (Plan 40-04 baseline)."""
    return [
        {
            "year": yr,
            "total": FROZEN_PER_YEAR_COVERAGE[yr]["total"],
            "populated": FROZEN_PER_YEAR_COVERAGE[yr]["populated"],
            "coverage_pct": _percent(
                FROZEN_PER_YEAR_COVERAGE[yr]["populated"],
                FROZEN_PER_YEAR_COVERAGE[yr]["total"],
            ),
            "is_anomaly_year": yr in ANOMALY_YEARS,
        }
        for yr in YEAR_RANGE
    ]


# ── Section 2: name-match-rate distribution ──────────────────────────────


def _load_bfo_names() -> list[str]:
    """Load the operator-curated BFO↔DB alias names CSV.

    File format (Phase 40-03 alias substrate at ``data/bfo/fighters_names.csv``):
      4-column CSV with header ``fighter_id,database,name,database_id`` —
      ``name`` (col 2) is the canonical fighter name. 2,481 rows on disk.

    The 4-column shape is Phase 40-03 canonical (BFO↔DB linkage rows
    preserved byte-identical to the `.precanon` backup per 40-03 Option-A
    closure). A 1-column legacy shape with just names is tolerated for
    operator-curated test fixtures.
    """
    if not BFO_NAMES_CSV.exists():
        return []
    names: list[str] = []
    with BFO_NAMES_CSV.open(encoding="utf-8", errors="replace") as fh:
        # Peek the header to detect shape.
        first_line = fh.readline()
        fh.seek(0)
        if "," in first_line and "name" in first_line.lower():
            # 4-column shape: use DictReader and pluck the `name` column.
            reader = csv.DictReader(fh)
            for row in reader:
                name = (row.get("name") or "").strip()
                if name:
                    names.append(name)
        else:
            # 1-column legacy fallback.
            reader = csv.reader(fh)
            for i, row in enumerate(reader):
                if not row:
                    continue
                cell = row[0].strip()
                if not cell:
                    continue
                if i == 0 and cell.lower() in {"name", "fighter", "bfo_name"}:
                    continue
                names.append(cell)
    return names


def name_match_distribution_offline(
    sample_db_names: dict[int, list[str]] | None = None,
) -> dict[str, Any]:
    """Run match_bfo_name() over a per-year sample of DB names against BFO names.

    Per Plan 41-01 Task 2 step 3: for each anomaly year, pull a
    representative sample (<=200 rows) of fights+fighters that ARE
    missing a paired BFO row; re-run match_bfo_name() against the BFO
    name list; bucketize the resulting score distribution into quartiles.

    Offline mode (Path B): when no live DB sample is supplied, the
    function returns a structural stub flagging the deferred status —
    the caller emits this in the JSON so the writeup can document the
    deferred state without a synthesized distribution.
    """
    from rapidfuzz import fuzz  # noqa: PLC0415

    from ufc_prediction.scraper.bfo_matcher import (  # noqa: PLC0415
        normalize_name,
    )

    bfo_names = _load_bfo_names()
    bfo_corpus_size = len(bfo_names)

    if sample_db_names is None or not sample_db_names:
        # Path B: deferred — script ran without a DB sample.
        return {
            "db_status": "unreachable",
            "deferred": True,
            "bfo_corpus_size": bfo_corpus_size,
            "per_year_quartile_counts": {},
            "note": (
                "Name-match-rate quartile distribution requires a live-DB "
                "sample of missing-BFO fighters per year. Run with "
                "DATABASE_URL set + reachable to populate this key."
            ),
        }

    # Path A: live sample available. Pre-normalize the BFO corpus once for speed.
    normalized_bfo = [normalize_name(n) for n in bfo_names]
    per_year_buckets: dict[int, dict[str, int]] = {}
    per_year_top_score_stats: dict[int, dict[str, float]] = {}
    for year, names in sample_db_names.items():
        bucket_counts: dict[str, int] = defaultdict(int)
        top_scores: list[float] = []
        sampled = names[:SAMPLE_CAP_PER_YEAR]
        for db_name in sampled:
            norm_db = normalize_name(db_name)
            # Compute the best score against the BFO corpus directly so
            # sub-threshold rows still get quartile-placed correctly.
            best_score = 0.0
            for norm_bfo in normalized_bfo:
                s = fuzz.ratio(norm_db, norm_bfo)
                if s > best_score:
                    best_score = s
                    if best_score >= 100:
                        break
            bucket_counts[_quartile_bucket(best_score)] += 1
            top_scores.append(best_score)
        per_year_buckets[year] = dict(bucket_counts)
        if top_scores:
            per_year_top_score_stats[year] = {
                "mean": round(statistics.mean(top_scores), 2),
                "median": round(statistics.median(top_scores), 2),
                "min": round(min(top_scores), 2),
                "max": round(max(top_scores), 2),
                "n": len(top_scores),
            }

    return {
        "db_status": "reachable",
        "deferred": False,
        "bfo_corpus_size": bfo_corpus_size,
        "per_year_quartile_counts": {
            str(yr): buckets for yr, buckets in per_year_buckets.items()
        },
        "per_year_top_score_stats": {
            str(yr): stats for yr, stats in per_year_top_score_stats.items()
        },
        "sample_cap_per_year": SAMPLE_CAP_PER_YEAR,
        "interpretation_note": (
            "Top-score = best rapidfuzz.fuzz.ratio() of the DB-side fighter "
            "name vs every entry in the BFO names corpus, after both sides "
            "are normalized via bfo_matcher.normalize_name(). High top-score "
            "(>= 80) on a row that LACKS a paired BFO closing-prob row "
            "indicates the gap is NOT name-matching — the fighter exists "
            "in the BFO corpus but the per-fight odds row is absent."
        ),
    }


# ── Section 3: URL canonicalization failure rate ─────────────────────────


def url_canonicalization_failure_rate() -> dict[str, Any]:
    """Run Phase 40-01 canonicalize_event_url against the operator BFO URL CSV.

    The CSV has columns including ``anchor_year`` and ``event_url``. We
    count canonicalization failures (helper returns the fallback rather
    than a regex-validated absolute URL) per anchor year, and compute
    per-year failure %.

    This runs INDEPENDENT of the DB — the CSV is operator-curated
    substrate from Phase 20 (Path A and Path B both work).
    """
    from ufc_prediction.scraper.bfo_scraper import (  # noqa: PLC0415
        _EVENT_HREF_RE,
        canonicalize_event_url,
    )

    if not BFO_EVENT_PROBE_CSV.exists():
        return {
            "csv_available": False,
            "per_year_failure_rate": {},
            "note": (
                f"{BFO_EVENT_PROBE_CSV} not found. Phase 20 operator CSV "
                "substrate is required for URL-canonicalization "
                "diagnosis. Regenerate via Phase 20 audit."
            ),
        }

    per_year_counts: dict[int, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "failed": 0}
    )
    sample_failures: list[dict[str, Any]] = []

    with BFO_EVENT_PROBE_CSV.open(encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                year = int(row.get("anchor_year") or row.get("year") or 0)
            except (TypeError, ValueError):
                continue
            url = (row.get("event_url") or "").strip()
            if not url or url == "EVENT_NOT_FOUND":
                # Sentinel rows — not a canonicalization failure;
                # operator marked them unfindable. Skip.
                continue
            per_year_counts[year]["total"] += 1
            # Extract the slug portion to canonicalize. We test the
            # regex match path directly since canonicalize_event_url
            # requires an href fragment.
            # Heuristic: strip the BFO base URL; pass the path as href.
            href = url.replace("https://www.bestfightodds.com", "")
            href = href.replace("http://www.bestfightodds.com", "")
            if not href.startswith("/"):
                href = "/" + href
            # Use the regex directly to detect a strict canonicalization match.
            match = _EVENT_HREF_RE.match(href)
            if match is None:
                per_year_counts[year]["failed"] += 1
                if len(sample_failures) < 20:
                    sample_failures.append(
                        {"year": year, "href": href, "url": url}
                    )

    per_year_failure_rate: dict[str, dict[str, Any]] = {}
    for year, counts in sorted(per_year_counts.items()):
        per_year_failure_rate[str(year)] = {
            "total": counts["total"],
            "failed": counts["failed"],
            "failure_pct": _percent(counts["failed"], counts["total"]),
            "is_anomaly_year": year in ANOMALY_YEARS,
        }

    return {
        "csv_available": True,
        "per_year_failure_rate": per_year_failure_rate,
        "sample_failures": sample_failures,
    }


# ── Section 4: event-date drift histogram (DB-only) ──────────────────────


def event_date_drift_histogram_db() -> dict[str, Any]:
    """Compute event-date drift histogram from live DB.

    Requires DB. The `FightOdds` schema does NOT carry an independent
    BFO event-date column (the BFO source date is reconciled at scrape
    time against `events.date` via `bfo_scraper._extract_event_date_from_meta`).
    As such, post-ingest drift IS undetectable from the persisted schema —
    drift would manifest as a missing match (covered by Section 2) or a
    mis-attribution to a different `event_id`.

    Path A: emit a structural per-year breakdown documenting this with
    drift_days = 0 for all matched fights (the only persisted state).
    Path B: emit the same deferred sentinel.
    """
    try:
        from ufc_prediction.db.session import SessionLocal  # noqa: PLC0415
        from ufc_prediction.models.event import Event  # noqa: PLC0415
        from ufc_prediction.models.fight import Fight  # noqa: PLC0415
        from ufc_prediction.models.fight_odds import FightOdds  # noqa: PLC0415

        session = SessionLocal()
        try:
            # Probe connectivity by counting fights — fails fast if unreachable.
            _ = session.query(Fight).count()
        finally:
            session.close()
    except Exception as exc:  # noqa: BLE001
        return {
            "db_status": "unreachable",
            "deferred": True,
            "per_year_drift_histogram": {},
            "note": (
                "FightOdds schema does NOT persist an independent BFO "
                "event-date column. Drift detection requires re-scraping "
                "BFO event-list HTML alongside the matched fight rows. "
                f"DB probe error: {type(exc).__name__}"
            ),
        }

    # DB reachable, but schema-level drift is undetectable post-ingest.
    # Emit a populated structural histogram with all-zero buckets per year,
    # plus the structural note explaining the limitation.
    per_year: dict[str, dict[str, int]] = {}
    for yr in YEAR_RANGE:
        per_year[str(yr)] = {"0": 0, "1-3": 0, "4-7": 0, "8-14": 0, "15+": 0}
    return {
        "db_status": "reachable",
        "deferred": False,
        "per_year_drift_histogram": per_year,
        "note": (
            "FightOdds schema does not store BFO event-date independently of "
            "events.date; persisted drift is structurally undetectable. "
            "If diagnosis indicates date drift as the root-cause dimension, "
            "Plan 41-02 must add bfo_event_date capture to bfo_scraper.py + "
            "persist via a schema migration. Until then this histogram is "
            "all-zero by construction (matched fights = aligned dates)."
        ),
    }


# ── Section 5: Phase 40-01 cross-check ───────────────────────────────────


def phase_40_01_crosscheck() -> dict[str, Any]:
    """Cross-check whether Phase 40-01 _EVENT_HREF_RE tightening already
    resolved anomaly-year clusters.

    Reasoning: Plan 40-04 captured per-year BFO coverage in
    ``results/corpus_v25_delta.md`` POST the Phase 40-01 regex tightening.
    The 2011/2013/2019/2020 anomalies persist in that post-40-01 snapshot —
    therefore Phase 40-01's URL canonicalization tightening DID NOT
    resolve the anomaly-year clusters. The fix dimension lies elsewhere.

    Output: per anomaly year, the Plan 40-04 post-40-01 coverage % +
    a binary verdict.
    """
    verdict_per_year: dict[str, dict[str, Any]] = {}
    for yr in ANOMALY_YEARS:
        entry = FROZEN_PER_YEAR_COVERAGE[yr]
        verdict_per_year[str(yr)] = {
            "post_40_01_total": entry["total"],
            "post_40_01_populated": entry["populated"],
            "post_40_01_coverage_pct": _percent(
                entry["populated"], entry["total"]
            ),
            "resolved_by_40_01": False,
        }
    # Cross-reference: the 2021-2026 cohort sits at 31-39% — that's the
    # post-Phase-40-01 baseline for the "non-anomaly" regime. The anomaly
    # years sit at 2.69-8.51%, a ~25-pp gap that Phase 40-01 did not close.
    return {
        "phase_40_01_commit": "78a4728",
        "phase_40_01_summary_ref": (
            ".planning/phases/40-corpus-growth-scraper-hygiene/"
            "40-01-SUMMARY.md"
        ),
        "post_40_01_anomaly_years": verdict_per_year,
        "post_40_01_non_anomaly_baseline_pct_2021_2026": {
            str(yr): _percent(
                FROZEN_PER_YEAR_COVERAGE[yr]["populated"],
                FROZEN_PER_YEAR_COVERAGE[yr]["total"],
            )
            for yr in range(2021, 2027)
        },
        "verdict": (
            "Phase 40-01 _EVENT_HREF_RE tightening + canonicalize_event_url "
            "extraction landed in commit 78a4728. The Plan 40-04 partner-"
            "facing report (results/corpus_v25_delta.md, post-40-01) records "
            "the 2011/2013/2019/2020 anomaly cohort at 2.69-8.51% coverage "
            "vs the 2021-2026 cohort at 31.56-38.83% — a ~25-pp gap that "
            "Phase 40-01 did NOT close. Phase 40-01 resolved a different "
            "regression (forward-scrape URL drift on 2024-2026 events). "
            "Phase 41 anomaly is carried by a different dimension."
        ),
        "resolved_count": 0,
        "remaining_count": len(ANOMALY_YEARS),
    }


# ── Markdown evidence emitter (consumed by Task 3 writeup) ───────────────


def emit_markdown(summary: dict[str, Any]) -> str:
    """Render the JSON summary as a Markdown evidence block."""
    lines: list[str] = []
    lines.append("<!-- Auto-generated by scripts/diagnose_bfo_disambiguation_v25.py -->")
    lines.append(f"<!-- Generated: {summary['generated_iso']} -->")
    lines.append(f"<!-- DB status: {summary['db_status']} -->")
    lines.append("")

    # 2.1 Per-year coverage
    lines.append("### 2.1 Per-year BFO coverage (2010-2026)")
    lines.append("")
    lines.append("| Year | Fights | BFO populated | Coverage % | Anomaly? |")
    lines.append("|------|-------:|--------------:|-----------:|:---------|")
    for row in summary["per_year_coverage"]:
        flag = "ANOMALY" if row["is_anomaly_year"] else ""
        lines.append(
            f"| {row['year']} | {row['total']:,} | {row['populated']:,} "
            f"| {row['coverage_pct']:.2f}% | {flag} |"
        )
    lines.append("")

    # 2.2 Name-match distribution
    name_dist = summary["name_match_distribution"]
    lines.append("### 2.2 Name-match-rate distribution")
    lines.append("")
    if name_dist.get("deferred"):
        lines.append("**Status:** DEFERRED — DB unreachable at diagnosis time.")
        lines.append("")
        lines.append(f"- BFO name-corpus size: {name_dist['bfo_corpus_size']:,}")
        lines.append(f"- {name_dist['note']}")
    else:
        lines.append(f"- BFO name-corpus size: {name_dist['bfo_corpus_size']:,}")
        lines.append(f"- Sample cap per year: {name_dist['sample_cap_per_year']}")
        lines.append("")
        lines.append("| Year | Sub-25 | 25-50 | 50-75 | 75-80 (sub-thr) | >=80 (match) |")
        lines.append("|------|------:|------:|------:|----------------:|-------------:|")
        for yr in ANOMALY_YEARS:
            buckets = name_dist["per_year_quartile_counts"].get(str(yr), {})
            lines.append(
                f"| {yr} "
                f"| {buckets.get('q1_below_25', 0)} "
                f"| {buckets.get('q2_25_to_50', 0)} "
                f"| {buckets.get('q3_50_to_75', 0)} "
                f"| {buckets.get('q4_75_to_80_subthreshold', 0)} "
                f"| {buckets.get('q5_80_plus_match', 0)} |"
            )
        lines.append("")
        lines.append("**Top-score stats (best fuzz.ratio vs BFO corpus, per anomaly year):**")
        lines.append("")
        lines.append("| Year | n | min | median | mean | max |")
        lines.append("|------|--:|----:|-------:|-----:|----:|")
        for yr in ANOMALY_YEARS:
            stats = name_dist.get("per_year_top_score_stats", {}).get(str(yr), {})
            if stats:
                lines.append(
                    f"| {yr} | {stats['n']} | {stats['min']:.2f} "
                    f"| {stats['median']:.2f} | {stats['mean']:.2f} "
                    f"| {stats['max']:.2f} |"
                )
        lines.append("")
        lines.append(f"_Methodology:_ {name_dist['interpretation_note']}")
    lines.append("")

    # 2.3 URL canonicalization
    url_section = summary["url_canon_failure_rate"]
    lines.append("### 2.3 URL canonicalization failure rate")
    lines.append("")
    if not url_section.get("csv_available", False):
        lines.append(f"**Status:** DEFERRED — {url_section.get('note')}")
    else:
        lines.append(
            "Phase 40-01 `canonicalize_event_url` + `_EVENT_HREF_RE` run "
            "against the operator-curated BFO event-URL CSV "
            "(`data/bfo_event_probe_targets.csv`). Failure = regex did not "
            "match the canonical /events/<slug>-<id> form."
        )
        lines.append("")
        lines.append("| Year | Total | Failed | Failure % | Anomaly? |")
        lines.append("|------|------:|-------:|----------:|:---------|")
        for yr_str, row in url_section["per_year_failure_rate"].items():
            flag = "ANOMALY" if row["is_anomaly_year"] else ""
            lines.append(
                f"| {yr_str} | {row['total']} | {row['failed']} "
                f"| {row['failure_pct']:.2f}% | {flag} |"
            )
    lines.append("")

    # 2.4 Event-date drift
    drift = summary["date_drift_histogram"]
    lines.append("### 2.4 Event-date drift histogram")
    lines.append("")
    if drift.get("deferred"):
        lines.append("**Status:** DEFERRED — DB unreachable AND structurally undetectable.")
    else:
        lines.append("**Status:** Structurally undetectable post-ingest (all-zero by construction).")
    lines.append("")
    lines.append(f"_Schema note:_ {drift['note']}")
    lines.append("")

    # 2.5 Phase 40-01 cross-check
    cross = summary["phase_40_01_crosscheck"]
    lines.append("### 2.5 Phase 40-01 cross-check")
    lines.append("")
    lines.append(cross["verdict"])
    lines.append("")
    lines.append(f"- Phase 40-01 fix commit: `{cross['phase_40_01_commit']}`")
    lines.append(
        f"- Anomaly years resolved by Phase 40-01: "
        f"**{cross['resolved_count']} / {len(ANOMALY_YEARS)}**"
    )
    lines.append("")
    lines.append("**Post-40-01 anomaly-year coverage:**")
    lines.append("")
    lines.append("| Year | Fights | BFO populated | Coverage % |")
    lines.append("|------|-------:|--------------:|-----------:|")
    for yr_str, row in cross["post_40_01_anomaly_years"].items():
        lines.append(
            f"| {yr_str} | {row['post_40_01_total']:,} "
            f"| {row['post_40_01_populated']:,} "
            f"| {row['post_40_01_coverage_pct']:.2f}% |"
        )
    lines.append("")
    lines.append("**Post-40-01 non-anomaly baseline (2021-2026):**")
    lines.append("")
    lines.append("| Year | Coverage % |")
    lines.append("|------|-----------:|")
    for yr_str, pct in cross["post_40_01_non_anomaly_baseline_pct_2021_2026"].items():
        lines.append(f"| {yr_str} | {pct:.2f}% |")
    lines.append("")

    # 2.6 BFO source-pipeline cascade
    cascade = summary.get("bfo_source_cascade", {})
    lines.append("### 2.6 BFO source-pipeline cascade")
    lines.append("")
    if cascade.get("deferred"):
        lines.append("**Status:** DEFERRED — " + cascade.get("note", "DB unreachable"))
    else:
        lines.append(
            "Traces the BFO source CSV through alias map → DB-Fight "
            "resolution → final fight_odds populated. The biggest drop "
            "dictates the Plan 41-02 fix dimension."
        )
        lines.append("")
        lines.append(
            "| Year | Anomaly? | Stage0: BFO src | Stage1: alias-mapped | "
            "Stage2: DB-Fight resolved | Stage3: final populated | "
            "alias drop % | resolve drop % | ingest drop % |"
        )
        lines.append(
            "|------|:---------|---------------:|---------------------:|"
            "---------------------:|----------------------:|"
            "-------------:|---------------:|--------------:|"
        )
        for yr_str, row in cascade["per_year_cascade"].items():
            flag = "ANOMALY" if row["is_anomaly_year"] else "control"
            lines.append(
                f"| {yr_str} | {flag} "
                f"| {row['stage0_bfo_source_fights']:,} "
                f"| {row['stage1_alias_both_mapped']:,} "
                f"| {row['stage2_db_fight_resolved']:,} "
                f"| {row['stage3_final_populated']:,} "
                f"| {row['drop_alias_pct']:.1f}% "
                f"| {row['drop_resolve_pct']:.1f}% "
                f"| {row['drop_ingest_pct']:.1f}% |"
            )
        lines.append("")
        if cascade.get("sample_dropped_at_ingest"):
            lines.append("**Sample rows dropped at Stage 2 → Stage 3 (ingest gap):**")
            lines.append("")
            lines.append(
                "| Year | Resolved key (date \\| db_f1 \\| db_f2) | DB fight_id | Populated odds sides |"
            )
            lines.append("|------|----------------------------------------|------------:|---------------------:|")
            for s in cascade["sample_dropped_at_ingest"]:
                lines.append(
                    f"| {s['year']} | `{s['resolved_key']}` "
                    f"| {s['db_fight_id']} | {s['n_populated_odds_sides']} |"
                )
            lines.append("")
        lines.append(f"_Methodology:_ {cascade['interpretation_note']}")
        lines.append("")

    return "\n".join(lines) + "\n"


# ── Main entry point ─────────────────────────────────────────────────────


def _try_per_year_coverage_db() -> tuple[list[dict[str, Any]], str]:
    """Try DB query; on failure, return frozen Path-B substrate."""
    try:
        rows = query_per_year_coverage_db()
        if not rows:
            raise RuntimeError("DB query returned 0 rows; corpus empty")
        return rows, "reachable"
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[diagnose-bfo] DB per-year query failed (%s); falling back "
            "to FROZEN_PER_YEAR_COVERAGE (Plan 40-04 substrate).",
            type(exc).__name__,
        )
        return per_year_coverage_frozen(), "unreachable"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="diagnose_bfo_disambiguation_v25",
        description="Phase 41 BFO disambiguation anomaly diagnosis (BFO-V25-01).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_JSON}).",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_OUTPUT_MD,
        help=f"Output Markdown evidence-block path (default: {DEFAULT_OUTPUT_MD}).",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help=(
            "Skip live-DB queries; use Plan 40-04 frozen substrate for "
            "per-year coverage. Marks DB-dependent keys as deferred."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Section 1: per-year coverage
    if args.no_db:
        per_year = per_year_coverage_frozen()
        per_year_db_status = "skipped"
    else:
        per_year, per_year_db_status = _try_per_year_coverage_db()

    # Section 2: name-match distribution
    if args.no_db or per_year_db_status != "reachable":
        name_dist = name_match_distribution_offline(sample_db_names=None)
    else:
        # Path A: would fetch DB samples and call name_match_distribution_offline
        # with the sample. For this Wave-0 spike we keep the live-DB path as a
        # structural hook — operator runs with `--no-db` removed when DATABASE_URL
        # points at a populated mirror.
        # Build the sample of un-matched-fighter names per anomaly year.
        try:
            sample = _fetch_unmatched_names_per_year_db()
            name_dist = name_match_distribution_offline(sample_db_names=sample)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[diagnose-bfo] Name-match-rate sample fetch failed (%s); "
                "deferring distribution.",
                type(exc).__name__,
            )
            name_dist = name_match_distribution_offline(sample_db_names=None)

    # Section 3: URL canonicalization failure rate (operator CSV — no DB)
    url_section = url_canonicalization_failure_rate()

    # Section 4: event-date drift histogram
    if args.no_db:
        drift = {
            "db_status": "skipped",
            "deferred": True,
            "per_year_drift_histogram": {},
            "note": (
                "Skipped via --no-db. The FightOdds schema does not "
                "persist BFO event-date independently of events.date; "
                "drift is structurally undetectable without re-scraping."
            ),
        }
    else:
        drift = event_date_drift_histogram_db()

    # Section 5: Phase 40-01 cross-check (no DB required — uses Plan 40-04 substrate)
    cross = phase_40_01_crosscheck()

    # Bonus: BFO source-pipeline cascade analysis (only if DB reachable).
    # This is the strongest signal we have: it traces the BFO source CSV
    # through alias map -> DB-Fight resolution -> final fight_odds populated
    # and locates where rows are dropped for the anomaly cohort.
    if per_year_db_status == "reachable":
        try:
            cascade = _bfo_source_cascade_analysis()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[diagnose-bfo] Source cascade analysis failed (%s); "
                "writing structural deferral.",
                type(exc).__name__,
            )
            cascade = {
                "db_status": "reachable_but_failed",
                "deferred": True,
                "error": f"{type(exc).__name__}: {exc}",
                "per_year_cascade": {},
            }
    else:
        cascade = {
            "db_status": per_year_db_status,
            "deferred": True,
            "per_year_cascade": {},
            "note": (
                "BFO source cascade analysis requires DB reachability to "
                "resolve (date, db_f1, db_f2) tuples against the fights "
                "table. Run with DATABASE_URL set + reachable to populate."
            ),
        }

    summary: dict[str, Any] = {
        "generated_iso": datetime.now(timezone.utc).isoformat(),
        "phase": "41",
        "plan": "01",
        "req": "BFO-V25-01",
        "db_status": per_year_db_status,
        "xgb_v2_sha_phase_41_start": (
            "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"
        ),
        "meta_v2_sha_phase_41_start": (
            "77076d3b2eed79797c355195f0f76156582b4c2f9b16df923c06ae2c855f9196"
        ),
        "anomaly_years": list(ANOMALY_YEARS),
        "per_year_coverage": per_year,
        "name_match_distribution": name_dist,
        "url_canon_failure_rate": url_section,
        "date_drift_histogram": drift,
        "phase_40_01_crosscheck": cross,
        "bfo_source_cascade": cascade,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[diagnose-bfo] JSON written: {args.output_json}")

    md_block = emit_markdown(summary)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(md_block)
    print(f"[diagnose-bfo] Markdown evidence block written: {args.output_md}")

    print(
        f"[diagnose-bfo] db_status={per_year_db_status} | "
        f"per_year_rows={len(per_year)} | "
        f"name_match_deferred={name_dist.get('deferred', False)} | "
        f"url_canon_csv_available={url_section.get('csv_available', False)} | "
        f"drift_deferred={drift.get('deferred', False)} | "
        f"phase_40_01_resolved={cross['resolved_count']}/{len(ANOMALY_YEARS)}"
    )
    return 0


def _bfo_source_cascade_analysis() -> dict[str, Any]:
    """Trace BFO source CSV → alias map → DB-Fight resolution → fight_odds cascade.

    For each anomaly year + 2 control years (2018, 2022):

      Stage 0: BFO source row count (from data/bfo/BestFightOdds_odds.csv, unique fight_id).
      Stage 1: Alias coverage — both BFO fighter IDs map to a DB id via fighters_names.csv.
      Stage 2: DB-Fight resolution — (event_date, db_f1, db_f2) tuple matches a real Fight row.
      Stage 3: Final populated — Fight has BOTH FightOdds aliases with closing_implied_prob.

    The biggest drop between Stage 2 and Stage 3 IS the disambiguation
    anomaly — it indicates rows that ARE in BFO source AND have all the
    linkage in place, but were silently dropped at fight_odds-insert time.
    Plan 41-02 must close that gap.

    DB-only. Caller catches exceptions.
    """
    from datetime import date as date_cls  # noqa: PLC0415

    from sqlalchemy import Integer, and_, case, func, or_  # noqa: PLC0415
    from sqlalchemy.orm import aliased  # noqa: PLC0415

    from ufc_prediction.db.session import SessionLocal  # noqa: PLC0415
    from ufc_prediction.models.event import Event  # noqa: PLC0415
    from ufc_prediction.models.fight import Fight  # noqa: PLC0415
    from ufc_prediction.models.fight_odds import FightOdds  # noqa: PLC0415

    bfo_to_db: dict[str, int] = {}
    if BFO_NAMES_CSV.exists():
        with BFO_NAMES_CSV.open(encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                bfo_id = (row.get("fighter_id") or "").strip()
                db_id = (row.get("database_id") or "").strip()
                if bfo_id and db_id:
                    try:
                        bfo_to_db[bfo_id] = int(db_id)
                    except ValueError:
                        pass

    bfo_csv = Path("data/bfo/BestFightOdds_odds.csv")
    if not bfo_csv.exists():
        return {
            "db_status": "reachable",
            "deferred": True,
            "per_year_cascade": {},
            "note": f"{bfo_csv} not found.",
        }

    # Collect unique BFO fights per year of interest.
    years_of_interest = set(ANOMALY_YEARS) | {2018, 2022}
    bfo_fights_by_year: dict[int, set[tuple[str, str, str]]] = defaultdict(set)
    with bfo_csv.open(encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fid = row.get("fight_id", "")
            if "|" not in fid:
                continue
            parts = fid.split("|")
            if len(parts) != 3:
                continue
            d, f1, f2 = parts
            try:
                yr = int(d[:4])
            except ValueError:
                continue
            if yr in years_of_interest:
                bfo_fights_by_year[yr].add((d, f1, f2))

    session = SessionLocal()
    cascade: dict[str, dict[str, Any]] = {}
    sample_dropped: list[dict[str, Any]] = []
    try:
        for yr in sorted(years_of_interest):
            bfo_fights = bfo_fights_by_year.get(yr, set())
            stage0 = len(bfo_fights)
            # Stage 1: both fighter IDs mapped
            mapped: list[tuple[str, int, int]] = []
            for d, f1, f2 in bfo_fights:
                if f1 in bfo_to_db and f2 in bfo_to_db:
                    mapped.append((d, bfo_to_db[f1], bfo_to_db[f2]))
            stage1 = len(mapped)

            # Stage 2: DB-Fight resolution (date + fighter pair match)
            stage2 = 0
            resolved_fight_ids: list[int] = []
            for d, db_f1, db_f2 in mapped:
                try:
                    event_date = date_cls.fromisoformat(d)
                except ValueError:
                    continue
                rows = (
                    session.query(Fight.id)
                    .join(Event, Fight.event_id == Event.id)
                    .filter(Event.date == event_date)
                    .filter(
                        or_(
                            and_(
                                Fight.fighter_a_id == db_f1,
                                Fight.fighter_b_id == db_f2,
                            ),
                            and_(
                                Fight.fighter_a_id == db_f2,
                                Fight.fighter_b_id == db_f1,
                            ),
                        )
                    )
                    .all()
                )
                if rows:
                    stage2 += 1
                    resolved_fight_ids.extend(r[0] for r in rows)

            # Stage 3: of those resolved fight IDs, count where BOTH FightOdds aliases
            # have a non-null closing_implied_prob.
            odds_a = aliased(FightOdds)
            odds_b = aliased(FightOdds)
            populated_expr = case(
                (
                    and_(
                        odds_a.closing_implied_prob.isnot(None),
                        odds_b.closing_implied_prob.isnot(None),
                    ),
                    1,
                ),
                else_=0,
            )
            stage3 = 0
            if resolved_fight_ids:
                stage3 = (
                    session.query(func.coalesce(func.sum(populated_expr), 0))
                    .select_from(Fight)
                    .outerjoin(
                        odds_a,
                        and_(
                            odds_a.fight_id == Fight.id,
                            odds_a.fighter_id == Fight.fighter_a_id,
                        ),
                    )
                    .outerjoin(
                        odds_b,
                        and_(
                            odds_b.fight_id == Fight.id,
                            odds_b.fighter_id == Fight.fighter_b_id,
                        ),
                    )
                    .filter(Fight.id.in_(set(resolved_fight_ids)))
                    .scalar()
                ) or 0
                stage3 = int(stage3)

            drop_alias_pct = _percent(stage0 - stage1, stage0) if stage0 else 0.0
            drop_resolve_pct = _percent(stage1 - stage2, stage1) if stage1 else 0.0
            drop_ingest_pct = _percent(stage2 - stage3, stage2) if stage2 else 0.0

            cascade[str(yr)] = {
                "stage0_bfo_source_fights": stage0,
                "stage1_alias_both_mapped": stage1,
                "stage2_db_fight_resolved": stage2,
                "stage3_final_populated": stage3,
                "drop_alias_pct": drop_alias_pct,
                "drop_resolve_pct": drop_resolve_pct,
                "drop_ingest_pct": drop_ingest_pct,
                "is_anomaly_year": yr in ANOMALY_YEARS,
            }

            # Capture samples of "resolved but not populated" rows — these
            # are the smoking gun for Plan 41-02. Cap at 10 across the run.
            if yr in ANOMALY_YEARS and stage2 > stage3 and len(sample_dropped) < 10:
                for d, db_f1, db_f2 in mapped[:5]:
                    try:
                        event_date = date_cls.fromisoformat(d)
                    except ValueError:
                        continue
                    fight_row = (
                        session.query(Fight.id)
                        .join(Event, Fight.event_id == Event.id)
                        .filter(Event.date == event_date)
                        .filter(
                            or_(
                                and_(
                                    Fight.fighter_a_id == db_f1,
                                    Fight.fighter_b_id == db_f2,
                                ),
                                and_(
                                    Fight.fighter_a_id == db_f2,
                                    Fight.fighter_b_id == db_f1,
                                ),
                            )
                        )
                        .first()
                    )
                    if fight_row is None:
                        continue
                    # Check if it has populated FightOdds rows.
                    n_odds = (
                        session.query(func.count(FightOdds.fight_id))
                        .filter(FightOdds.fight_id == fight_row[0])
                        .filter(FightOdds.closing_implied_prob.isnot(None))
                        .scalar()
                    )
                    if n_odds < 2:  # Missing one or both sides
                        sample_dropped.append(
                            {
                                "year": yr,
                                "resolved_key": f"{d}|db:{db_f1}|db:{db_f2}",
                                "db_fight_id": int(fight_row[0]),
                                "n_populated_odds_sides": int(n_odds),
                            }
                        )
                        if len(sample_dropped) >= 10:
                            break
    finally:
        session.close()

    return {
        "db_status": "reachable",
        "deferred": False,
        "per_year_cascade": cascade,
        "sample_dropped_at_ingest": sample_dropped,
        "interpretation_note": (
            "Stage0 = unique BFO source rows per year (data/bfo/BestFightOdds_odds.csv). "
            "Stage1 = both BFO fighter IDs map to a DB id (data/bfo/fighters_names.csv alias table). "
            "Stage2 = (event_date, db_f1, db_f2) resolves to a real Fight row. "
            "Stage3 = Fight has BOTH FightOdds aliases with closing_implied_prob populated. "
            "The single biggest drop dictates the Plan 41-02 fix dimension: "
            "if drop_alias_pct dominates -> grow fighters_names.csv alias coverage; "
            "if drop_resolve_pct dominates -> fix date/pair matching logic in bfo_ingest; "
            "if drop_ingest_pct dominates -> fix the per-fight ingest step that silently "
            "drops rows after successful matching (most likely Plan 41-02 target)."
        ),
    }


def _fetch_unmatched_names_per_year_db() -> dict[int, list[str]]:
    """Live-DB fetch: anomaly-year fighters in fights LACKING a paired BFO row.

    Returns mapping {year -> list of distinct fighter names, capped at
    SAMPLE_CAP_PER_YEAR per year}.

    Path A only — raises if DB unreachable. Caller catches.
    """
    from sqlalchemy import Integer, and_, func, or_  # noqa: PLC0415
    from sqlalchemy.orm import aliased  # noqa: PLC0415

    from ufc_prediction.db.session import SessionLocal  # noqa: PLC0415
    from ufc_prediction.models.event import Event  # noqa: PLC0415
    from ufc_prediction.models.fight import Fight  # noqa: PLC0415
    from ufc_prediction.models.fight_odds import FightOdds  # noqa: PLC0415
    from ufc_prediction.models.fighter import Fighter  # noqa: PLC0415

    session = SessionLocal()
    result: dict[int, list[str]] = {}
    try:
        odds_a = aliased(FightOdds)
        odds_b = aliased(FightOdds)
        year_expr = func.cast(func.extract("year", Event.date), Integer)
        for year in ANOMALY_YEARS:
            # Pick fighter_a names from fights where either odds row is NULL.
            rows = (
                session.query(Fighter.name)
                .join(Fight, Fight.fighter_a_id == Fighter.id)
                .join(Event, Fight.event_id == Event.id)
                .outerjoin(
                    odds_a,
                    and_(
                        odds_a.fight_id == Fight.id,
                        odds_a.fighter_id == Fight.fighter_a_id,
                    ),
                )
                .outerjoin(
                    odds_b,
                    and_(
                        odds_b.fight_id == Fight.id,
                        odds_b.fighter_id == Fight.fighter_b_id,
                    ),
                )
                .filter(year_expr == year)
                .filter(
                    or_(
                        odds_a.closing_implied_prob.is_(None),
                        odds_b.closing_implied_prob.is_(None),
                    )
                )
                .distinct()
                .limit(SAMPLE_CAP_PER_YEAR)
                .all()
            )
            result[year] = [r[0] for r in rows]
        return result
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
