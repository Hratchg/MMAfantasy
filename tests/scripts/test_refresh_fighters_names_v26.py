"""Phase 49 HYG-V26-03 — minimal smoke test for refresh_fighters_names_v26.py.

Verifies the dry-run path runs end-to-end without DB access and that the
key invariants of the classifier (additive new rows vs conflict gating)
hold for synthetic fixtures.

The production `--apply` path requires Supabase access and live BFO
connectivity (backlogged to v2.7+). That path is NOT tested here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "refresh_fighters_names_v26.py"


@pytest.fixture(scope="module")
def mod():
    """Import the script via importlib (it's a script, not a package)."""
    spec = importlib.util.spec_from_file_location(
        "refresh_fighters_names_v26",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules["refresh_fighters_names_v26"] = m
    spec.loader.exec_module(m)
    return m


def test_dry_run_runs_end_to_end_without_db(mod, capsys) -> None:
    """`--dry-run` mode must complete without touching DB / live BFO."""
    rc = mod.main(["--dry-run"])
    captured = capsys.readouterr()
    assert rc in (0, 1), f"unexpected exit code {rc}"
    assert "DRY-RUN merge summary" in captured.out


def test_classify_emits_conflict_when_same_name_different_bfo_id(mod) -> None:
    """Same DB-name + different BFO numeric id -> conflict gate fires."""
    baseline = [
        mod.CSVRow(
            fighter_id="819",
            database="ufcstats",
            name="Jon Jones",
            database_id="42",
        ),
    ]
    # New match against the same fighter name surfaces with a DIFFERENT
    # BFO numeric id (e.g., BFO renumbered or the matcher picked a
    # different /fighters/<slug>-<id>).
    drifted_match = mod.MatchResult(
        db_fighter=mod.DBFighter(id=42, name="Jon Jones", source="ufcstats"),
        matched_url="https://www.bestfightodds.com/fighters/Jon-Jones-999",
        bfo_numeric_id="999",
    )
    new_rows, conflicts = mod._classify([drifted_match], baseline)
    assert len(new_rows) == 0
    assert len(conflicts) == 1
    assert conflicts[0].fighter_id == "999"


def test_classify_additive_for_new_fighter(mod) -> None:
    """A genuinely new DB fighter not in the baseline -> additive row."""
    baseline = [
        mod.CSVRow(
            fighter_id="819",
            database="ufcstats",
            name="Jon Jones",
            database_id="42",
        ),
    ]
    new_match = mod.MatchResult(
        db_fighter=mod.DBFighter(id=43, name="Some New Fighter", source="ufcstats"),
        matched_url="https://www.bestfightodds.com/fighters/Some-New-Fighter-1234",
        bfo_numeric_id="1234",
    )
    new_rows, conflicts = mod._classify([new_match], baseline)
    assert len(new_rows) == 1
    assert len(conflicts) == 0
    assert new_rows[0].fighter_id == "1234"
    assert new_rows[0].database_id == "43"


def test_classify_no_op_for_reconfirm(mod) -> None:
    """Same name + same BFO id -> re-confirm, neither new nor conflict."""
    baseline = [
        mod.CSVRow(
            fighter_id="819",
            database="ufcstats",
            name="Jon Jones",
            database_id="42",
        ),
    ]
    reconfirm = mod.MatchResult(
        db_fighter=mod.DBFighter(id=42, name="Jon Jones", source="ufcstats"),
        matched_url="https://www.bestfightodds.com/fighters/Jon-Jones-819",
        bfo_numeric_id="819",
    )
    new_rows, conflicts = mod._classify([reconfirm], baseline)
    assert new_rows == []
    assert conflicts == []


def test_baseline_below_phase_28_04_floor_raises(mod, tmp_path) -> None:
    """Reading a baseline with < 399 rows must halt the additive invariant."""
    # 5 rows; below the floor.
    csv_path = tmp_path / "fighters_names.csv"
    csv_path.write_text(
        "fighter_id,database,name,database_id\n"
        + "\n".join(f"{i},ufcstats,F{i},{i}" for i in range(5))
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="additive-merge invariant violated"):
        mod._read_csv_baseline(csv_path)


def test_extract_bfo_numeric_id(mod) -> None:
    """Numeric id extraction round-trip."""
    assert (
        mod._extract_bfo_numeric_id("https://www.bestfightodds.com/fighters/Jon-Jones-819") == "819"
    )
    assert mod._extract_bfo_numeric_id("https://other.example/fighters/Jon-Jones-819") is None
    assert mod._extract_bfo_numeric_id("https://www.bestfightodds.com/fighters/foo--123") is None, (
        "trailing-dash slug must reject (WR-01 / CORPUS-V25-02 carry-forward)"
    )
