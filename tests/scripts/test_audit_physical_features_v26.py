"""Phase 57 FEAT-V26-03 — audit-driver unit tests."""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "audit_physical_features_v26.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location(
        "audit_physical_features_v26", SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules["audit_physical_features_v26"] = m
    spec.loader.exec_module(m)
    return m


def test_era_bucket_assignment(mod) -> None:
    assert mod._era_bucket(2008) == "pre-2010"
    assert mod._era_bucket(2014) == "2010-2014"
    assert mod._era_bucket(2016) == "2015-2019"
    assert mod._era_bucket(2026) == "2020-2026"


def test_is_missing_handles_none_and_nan(mod) -> None:
    import math
    assert mod._is_missing(None) is True
    assert mod._is_missing(float("nan")) is True
    assert mod._is_missing(math.nan) is True
    assert mod._is_missing(0.0) is False
    assert mod._is_missing("") is False  # not a numeric NaN


def test_audit_missingness_synthetic(mod) -> None:
    cohort = mod.CohortBucket(division="Heavy", era="2020-2026")
    fighters = [
        {"height_inches": 72.0, "reach_inches": 72.0, "leg_reach_inches": None,
         "date_of_birth": date(1990, 1, 1)},
        {"height_inches": 73.0, "reach_inches": 73.0, "leg_reach_inches": 40.0,
         "date_of_birth": None},
        {"height_inches": None, "reach_inches": 71.0, "leg_reach_inches": 39.0,
         "date_of_birth": date(1985, 1, 1)},
    ]
    result = mod.audit_missingness({cohort: fighters})
    assert result.n_fighters_total == 3
    assert result.n_divisions == 1
    # Per-column lookups
    by_col_cohort = {(r.column, r.cohort.label()): r for r in result.rows}
    assert by_col_cohort[("height_inches", "Heavy|2020-2026")].n_missing == 1
    assert by_col_cohort[("leg_reach_inches", "Heavy|2020-2026")].n_missing == 1
    assert by_col_cohort[("date_of_birth", "Heavy|2020-2026")].n_missing == 1


def test_detect_systematic_bias_threshold_logic(mod) -> None:
    # Build an audit where leg_reach_inches has high missingness in 3 of 5
    # cohorts -> 60% > default 30% threshold -> bias_detected=True
    rows = []
    summary = {col: {} for col in mod.PHYSICAL_COLUMNS}
    for i in range(5):
        cohort = mod.CohortBucket(division=f"D{i}", era="2020-2026")
        n = 10
        n_missing = 8 if i < 3 else 1  # 80% missing in first 3, 10% in last 2
        rows.append(
            mod.ColumnMissingnessRow(
                cohort=cohort, column="leg_reach_inches",
                n_fighters=n, n_missing=n_missing,
                pct_missing=n_missing / n,
            )
        )
        summary["leg_reach_inches"][cohort.label()] = n_missing / n
        for other in mod.PHYSICAL_COLUMNS:
            if other == "leg_reach_inches":
                continue
            summary[other][cohort.label()] = 0.0  # clean

    audit = mod.AuditResult(
        rows=rows,
        summary_by_column=summary,
        n_fighters_total=50,
        n_divisions=5,
    )
    findings = mod.detect_systematic_bias(audit)
    assert findings["leg_reach_inches"]["bias_detected"] is True
    assert findings["height_inches"]["bias_detected"] is False
    assert findings["reach_inches"]["bias_detected"] is False


def test_dry_run_main_runs_end_to_end(mod, tmp_path, capsys) -> None:
    out = tmp_path / "findings.md"
    rc = mod.main(["--dry-run", "--out", str(out)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Audit complete (dry-run)" in captured.out
    assert out.exists()
    body = out.read_text(encoding="utf-8")
    assert "Physical-Feature Bias Audit" in body
    assert "## Per-Column Findings" in body
    assert "## Recommendation" in body


def test_apply_mode_requires_database_url(mod) -> None:
    rc = mod.main(["--apply"])
    assert rc == 2  # missing database_url -> error exit
