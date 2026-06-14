"""Phase 57 FEAT-V26-03 — physical-feature bias audit driver.

Runs a per-column-missingness × division × era audit on:
  - reach_diff (Fighter.reach_inches)
  - height_diff (Fighter.height_inches)
  - leg_reach_diff (Fighter.leg_reach_inches)
  - age_diff (Fighter.date_of_birth -> age at fight)

Plus secondary physical columns where present.

Modes:
  --dry-run  Use synthetic fixture data; no DB. Validates the audit
             pipeline and produces a placeholder findings doc.
  --apply    Run against the live DB (Supabase via DATABASE_URL).
             Writes results/physical_features_bias_audit_v26.md with
             real numbers.

Output: results/physical_features_bias_audit_v26.md per repository
convention (Phase 41 / 42 / 45 precedent for spike-findings writeups).

NO MODEL TOUCHES. AUDIT-01 invariant: xgb_v2 + meta_v2 SHAs UNCHANGED.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "results" / "physical_features_bias_audit_v26.md"

# Era buckets per Phase 18 / 19 / 20 cohort-comparability precedent.
ERA_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("pre-2010", 0, 2009),
    ("2010-2014", 2010, 2014),
    ("2015-2019", 2015, 2019),
    ("2020-2026", 2020, 2026),
)

# Physical feature columns audited.
PHYSICAL_COLUMNS: tuple[str, ...] = (
    "height_inches",
    "reach_inches",
    "leg_reach_inches",
    "date_of_birth",
)


@dataclass(frozen=True)
class CohortBucket:
    """One (division, era) cohort."""

    division: str
    era: str

    def label(self) -> str:
        return f"{self.division}|{self.era}"


@dataclass
class ColumnMissingnessRow:
    """Per-cohort per-column missingness row."""

    cohort: CohortBucket
    column: str
    n_fighters: int
    n_missing: int
    pct_missing: float


@dataclass
class AuditResult:
    """Aggregate audit output."""

    rows: list[ColumnMissingnessRow] = field(default_factory=list)
    summary_by_column: dict[str, dict[str, float]] = field(default_factory=dict)
    n_fighters_total: int = 0
    n_divisions: int = 0
    n_eras: int = len(ERA_BUCKETS)


# ── Cohort assignment helpers ────────────────────────────────────────────


def _era_bucket(year: int) -> str:
    for label, lo, hi in ERA_BUCKETS:
        if lo <= year <= hi:
            return label
    return "unknown"


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        import math
        return math.isnan(value)
    return False


# ── Audit core ────────────────────────────────────────────────────────────


def audit_missingness(
    fighters_by_cohort: dict[CohortBucket, list[dict[str, object]]],
) -> AuditResult:
    """Compute per-cohort missingness for each PHYSICAL_COLUMN.

    Args:
        fighters_by_cohort: {cohort -> [fighter_row_dict, ...]}.
            Each row dict carries the PHYSICAL_COLUMNS keys.

    Returns:
        AuditResult with per-cohort rows + per-column summary.
    """
    rows: list[ColumnMissingnessRow] = []
    summary: dict[str, dict[str, float]] = {col: {} for col in PHYSICAL_COLUMNS}
    n_total = 0
    divisions: set[str] = set()
    for cohort, fighter_rows in fighters_by_cohort.items():
        divisions.add(cohort.division)
        n_total += len(fighter_rows)
        for col in PHYSICAL_COLUMNS:
            n = len(fighter_rows)
            n_missing = sum(1 for r in fighter_rows if _is_missing(r.get(col)))
            pct = (n_missing / n) if n > 0 else 0.0
            rows.append(
                ColumnMissingnessRow(
                    cohort=cohort,
                    column=col,
                    n_fighters=n,
                    n_missing=n_missing,
                    pct_missing=pct,
                )
            )
            summary[col][cohort.label()] = pct
    return AuditResult(
        rows=rows,
        summary_by_column=summary,
        n_fighters_total=n_total,
        n_divisions=len(divisions),
    )


# ── Bias-detection heuristic ──────────────────────────────────────────────


def detect_systematic_bias(
    audit: AuditResult,
    *,
    pct_threshold: float = 0.25,
    cohort_share_threshold: float = 0.30,
) -> dict[str, dict[str, object]]:
    """Flag columns where >cohort_share_threshold of cohorts exceed pct_threshold.

    Heuristic: if more than 30% of cohorts have >25% missingness on a
    column, that column has systematic bias risk. Tunable via args; the
    defaults reflect Phase 57 CONTEXT decisions.
    """
    findings: dict[str, dict[str, object]] = {}
    for col in PHYSICAL_COLUMNS:
        cohorts = audit.summary_by_column[col]
        if not cohorts:
            continue
        n_cohorts = len(cohorts)
        n_above_threshold = sum(
            1 for pct in cohorts.values() if pct > pct_threshold
        )
        cohort_share = n_above_threshold / n_cohorts if n_cohorts > 0 else 0.0
        bias_detected = cohort_share > cohort_share_threshold
        max_missing_cohort = (
            max(cohorts.items(), key=lambda kv: kv[1])
            if cohorts
            else (None, 0.0)
        )
        findings[col] = {
            "bias_detected": bias_detected,
            "n_cohorts_total": n_cohorts,
            "n_cohorts_above_threshold": n_above_threshold,
            "max_missing_cohort": max_missing_cohort[0],
            "max_missing_pct": max_missing_cohort[1],
        }
    return findings


# ── Results doc emit ──────────────────────────────────────────────────────


def emit_findings_md(
    audit: AuditResult,
    findings: dict[str, dict[str, object]],
    *,
    out_path: Path = RESULTS_PATH,
    mode_label: str = "dry-run",
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Physical-Feature Bias Audit — v2.6 (Phase 57)",
        "",
        f"**Mode:** {mode_label}",
        "**Authored:** Phase 57 FEAT-V26-03",
        "**AUDIT-01 invariant:** xgb_v2 + meta_v2 byte-identical (no model touch)",
        "",
        "## Methodology",
        "",
        "Per-(division, era) cohort missingness audit on physical-feature ",
        "columns: `height_inches`, `reach_inches`, `leg_reach_inches`, ",
        "`date_of_birth`. Era buckets per Phase 18 / 19 / 20 cohort-",
        "comparability precedent.",
        "",
        "Bias-detection heuristic: a column has systematic bias risk if ",
        "more than 30% of cohorts exhibit >25% missingness. Tunable; ",
        "defaults reflect Phase 57 CONTEXT decisions.",
        "",
        "## Audit Scope",
        "",
        f"- Total fighters audited: {audit.n_fighters_total}",
        f"- Divisions: {audit.n_divisions}",
        f"- Eras: {audit.n_eras} ({', '.join(b[0] for b in ERA_BUCKETS)})",
        "",
        "## Per-Column Findings",
        "",
        "| Column | Bias detected | N cohorts | Cohorts >25% missing | "
        "Worst cohort | Worst pct |",
        "|---|---|---|---|---|---|",
    ]
    for col in PHYSICAL_COLUMNS:
        f = findings.get(col)
        if f is None:
            continue
        flag = "YES" if f["bias_detected"] else "no"
        lines.append(
            f"| `{col}` | {flag} | {f['n_cohorts_total']} | "
            f"{f['n_cohorts_above_threshold']} | "
            f"{f['max_missing_cohort']} | "
            f"{f['max_missing_pct']:.1%} |"
        )

    any_bias = any(f["bias_detected"] for f in findings.values())
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "- **Per-cohort NaN handling action needed:** "
            + ("**YES** — backlog as FEAT-V26-03 in v2.7+." if any_bias else "no."),
        ]
    )
    if any_bias:
        biased = [c for c, f in findings.items() if f["bias_detected"]]
        lines.append(
            f"- Biased columns: {', '.join(f'`{c}`' for c in biased)}"
        )
        lines.append(
            "- Recommended fix: per-cohort imputation OR cohort-stratified "
            "feature engineering (FEAT-V26-03; v2.7+ scope per Phase 57 "
            "CONTEXT.md backlog decision)."
        )
    else:
        lines.append(
            "- Closed as documented negative result; no FEAT-V26-03 action."
        )

    if mode_label == "dry-run":
        lines.extend(
            [
                "",
                "## Note",
                "",
                "This is a dry-run output against synthetic fixture data. "
                "v2.6.1 follow-on re-runs against the production Supabase "
                "corpus to produce the canonical findings.",
            ]
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ── Dry-run fixture ───────────────────────────────────────────────────────


def _dry_run_fixture() -> dict[CohortBucket, list[dict[str, object]]]:
    """Synthetic fighters across 2 divisions × 4 eras for dry-run smoke testing."""
    fixture: dict[CohortBucket, list[dict[str, object]]] = defaultdict(list)
    for division in ("Lightweight", "Heavyweight"):
        for era_label, _, _ in ERA_BUCKETS:
            cohort = CohortBucket(division=division, era=era_label)
            # 10 fighters per cohort, missingness scaled by era + division
            for i in range(10):
                # Simulate older eras missing leg_reach more often
                leg_reach_missing = era_label == "pre-2010" and i < 8
                fighter = {
                    "height_inches": 70.0 + i,
                    "reach_inches": 70.0 + i,
                    "leg_reach_inches": None if leg_reach_missing else 40.0 + i,
                    "date_of_birth": date(1985, 1, 1) if i % 5 == 0 else None,
                }
                fixture[cohort].append(fighter)
    return dict(fixture)


# ── Live-DB loader (apply mode) ───────────────────────────────────────────


def _load_db_fighter_cohorts(
    database_url: str,
) -> dict[CohortBucket, list[dict[str, object]]]:
    """Load DB fighters grouped by (division, era).

    Cohort assignment: division from `Fight.weight_class`; era from
    `Fight.event.date` year. A fighter participating across multiple
    eras/divisions appears in MULTIPLE cohorts (intentional — the audit
    measures missingness per cohort, not per fighter).
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from ufc_prediction.models.event import Event
    from ufc_prediction.models.fight import Fight
    from ufc_prediction.models.fighter import Fighter

    engine = create_engine(database_url)
    cohorts: dict[CohortBucket, list[dict[str, object]]] = defaultdict(list)
    with Session(engine) as session:
        fights = session.query(Fight).join(Event).all()
        for fight in fights:
            event_year = fight.event.date.year
            era = _era_bucket(event_year)
            weight_class = fight.weight_class or "Unknown"
            for fighter_id in (fight.fighter_a_id, fight.fighter_b_id):
                fighter = session.get(Fighter, fighter_id)
                if fighter is None:
                    continue
                cohort = CohortBucket(division=weight_class, era=era)
                cohorts[cohort].append(
                    {
                        "height_inches": fighter.height_inches,
                        "reach_inches": fighter.reach_inches,
                        "leg_reach_inches": fighter.leg_reach_inches,
                        "date_of_birth": fighter.date_of_birth,
                    }
                )
    return dict(cohorts)


# ── Driver ────────────────────────────────────────────────────────────────


def run(
    *,
    dry_run: bool,
    database_url: Optional[str] = None,
    out_path: Path = RESULTS_PATH,
) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if dry_run:
        fixture = _dry_run_fixture()
        logger.info("Dry-run: %d cohorts loaded from fixture", len(fixture))
        audit = audit_missingness(fixture)
        findings = detect_systematic_bias(audit)
        emit_findings_md(audit, findings, out_path=out_path, mode_label="dry-run")
        sys.stdout.write(
            f"Audit complete (dry-run). "
            f"Findings written to {out_path}.\n"
        )
        return 0

    if database_url is None:
        logger.error("--apply requires --database-url or DATABASE_URL env var")
        return 2
    cohorts = _load_db_fighter_cohorts(database_url)
    logger.info("Apply: %d cohorts loaded from DB", len(cohorts))
    audit = audit_missingness(cohorts)
    findings = detect_systematic_bias(audit)
    emit_findings_md(audit, findings, out_path=out_path, mode_label="live-db")
    sys.stdout.write(
        f"Audit complete (live-db). Findings written to {out_path}.\n"
    )
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dry-run", action="store_true")
    grp.add_argument("--apply", action="store_true")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--out", default=str(RESULTS_PATH))
    args = parser.parse_args(argv)
    return run(
        dry_run=args.dry_run,
        database_url=args.database_url,
        out_path=Path(args.out),
    )


if __name__ == "__main__":
    raise SystemExit(main())
