"""`ufc elo compute` fails closed when the Sherdog debutant-seed CSV is missing.

Without ``data/sherdog/pre_ufc_records.csv`` the engine falls back to flat-1500
debutant ratings, and ``flush_snapshots`` would then delete and rewrite every
overall ``elo_snapshots`` row in the shared DB with that degraded substrate.
The command must refuse before opening a DB session unless the operator
passes ``--allow-unseeded`` for an intentional flat-1500 run.

No DB and no real ``elo compute``: every DB touchpoint is monkeypatched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from ufc_prediction.cli import main as cli_main
from ufc_prediction.cli.main import app

runner = CliRunner()

_CSV_HEADER = "fighter_id,n_pre_ufc_fights,win_rate,org_tier\n"


def _forbid_db(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Make every DB touchpoint of compute_elo record that it was called."""
    calls = {"session": 0, "flush": 0}

    def _session() -> Any:
        calls["session"] += 1
        return MagicMock()

    def _flush(*_a: Any, **_k: Any) -> int:
        calls["flush"] += 1
        return 0

    monkeypatch.setattr(cli_main, "SessionLocal", _session)
    monkeypatch.setattr(cli_main, "flush_snapshots", _flush)
    monkeypatch.setattr(cli_main, "flush_domain_snapshots", _flush)
    monkeypatch.setattr(cli_main, "load_fights_chronological", lambda _s: [])
    return calls


def test_missing_seed_csv_exits_1_without_touching_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_main, "_SHERDOG_PRE_UFC_CSV", tmp_path / "missing.csv")
    calls = _forbid_db(monkeypatch)

    result = runner.invoke(app, ["elo", "compute", "--no-domain"])

    assert result.exit_code == 1, result.output
    assert "--allow-unseeded" in result.output
    assert calls == {"session": 0, "flush": 0}


def test_header_only_seed_csv_exits_1_without_touching_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv = tmp_path / "pre_ufc_records.csv"
    csv.write_text(_CSV_HEADER)
    monkeypatch.setattr(cli_main, "_SHERDOG_PRE_UFC_CSV", csv)
    calls = _forbid_db(monkeypatch)

    result = runner.invoke(app, ["elo", "compute", "--no-domain"])

    assert result.exit_code == 1, result.output
    assert calls == {"session": 0, "flush": 0}


def test_malformed_seed_csv_exits_1_without_touching_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv = tmp_path / "pre_ufc_records.csv"
    csv.write_text(_CSV_HEADER + "1,,0.5,major\n")
    monkeypatch.setattr(cli_main, "_SHERDOG_PRE_UFC_CSV", csv)
    calls = _forbid_db(monkeypatch)

    result = runner.invoke(app, ["elo", "compute", "--no-domain"])

    assert result.exit_code == 1, result.output
    assert calls == {"session": 0, "flush": 0}


def test_allow_unseeded_runs_flat_1500_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_main, "_SHERDOG_PRE_UFC_CSV", tmp_path / "missing.csv")
    calls = _forbid_db(monkeypatch)

    result = runner.invoke(app, ["elo", "compute", "--no-domain", "--allow-unseeded"])

    assert result.exit_code == 0, result.output
    assert calls == {"session": 1, "flush": 1}


def test_present_seed_csv_runs_normally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv = tmp_path / "pre_ufc_records.csv"
    csv.write_text(_CSV_HEADER + "1,10,0.8,major\n")
    monkeypatch.setattr(cli_main, "_SHERDOG_PRE_UFC_CSV", csv)
    calls = _forbid_db(monkeypatch)

    result = runner.invoke(app, ["elo", "compute", "--no-domain"])

    assert result.exit_code == 0, result.output
    assert "Loaded 1 debutant Elo seeds" in result.output
    assert calls == {"session": 1, "flush": 1}
