"""Integration tests for the CLI export sub-commands (fighters, rankings, history)."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from ufc_prediction.cli import export as export_module
from ufc_prediction.cli.main import app

runner = CliRunner()


@pytest.fixture()
def patch_export_session(session, seed_cli_data, monkeypatch):
    """Monkeypatch SessionLocal in both main and export modules."""
    monkeypatch.setattr("ufc_prediction.cli.main.SessionLocal", lambda: session)
    monkeypatch.setattr(export_module, "SessionLocal", lambda: session)
    return seed_cli_data


# ── Fighters Export ──────────────────────────────────────────────────────────


def test_export_fighters_csv(patch_export_session):
    """Export all fighters as CSV to stdout."""
    result = runner.invoke(app, ["export", "fighters", "--format", "csv"])
    assert result.exit_code == 0
    lines = result.stdout.strip().split("\n")
    assert len(lines) >= 2  # header + at least 1 data row
    header = lines[0]
    assert "name" in header
    assert "elo" in header
    assert "division" in header


def test_export_fighters_json(patch_export_session):
    """Export all fighters as JSON to stdout."""
    result = runner.invoke(app, ["export", "fighters", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert isinstance(data, list)
    assert len(data) >= 1
    assert "name" in data[0]
    assert "elo" in data[0]


def test_export_fighters_to_file(patch_export_session, tmp_path):
    """Export fighters to a file using --output."""
    outfile = tmp_path / "fighters.csv"
    result = runner.invoke(
        app, ["export", "fighters", "--format", "csv", "--output", str(outfile)]
    )
    assert result.exit_code == 0
    assert outfile.exists()
    content = outfile.read_text()
    assert "name" in content
    assert "elo" in content


# ── Rankings Export ──────────────────────────────────────────────────────────


def test_export_rankings_csv(patch_export_session):
    """Export Lightweight rankings as CSV."""
    result = runner.invoke(
        app, ["export", "rankings", "Lightweight", "--format", "csv"]
    )
    assert result.exit_code == 0
    lines = result.stdout.strip().split("\n")
    assert len(lines) >= 2  # header + data
    assert "name" in lines[0]
    assert "elo" in lines[0]


def test_export_rankings_json(patch_export_session):
    """Export Lightweight rankings as JSON."""
    result = runner.invoke(
        app, ["export", "rankings", "Lightweight", "--format", "json"]
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert isinstance(data, list)
    assert len(data) >= 1


def test_export_rankings_invalid_division(patch_export_session):
    """Export with invalid division shows error."""
    result = runner.invoke(app, ["export", "rankings", "Nonexistent"])
    assert result.exit_code != 0


# ── History Export ───────────────────────────────────────────────────────────


def test_export_history_csv(patch_export_session):
    """Export Khabib's Elo history as CSV."""
    result = runner.invoke(
        app, ["export", "history", "Khabib", "--format", "csv"]
    )
    assert result.exit_code == 0
    lines = result.stdout.strip().split("\n")
    assert len(lines) >= 3  # header + 2 fights for Khabib
    assert "fight_date" in lines[0]
    assert "elo_after" in lines[0]


def test_export_history_json(patch_export_session):
    """Export Khabib's Elo history as JSON."""
    result = runner.invoke(
        app, ["export", "history", "Khabib", "--format", "json"]
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert isinstance(data, list)
    assert len(data) >= 2  # Khabib has 2 Elo snapshots in seed data


def test_export_history_not_found(patch_export_session):
    """Export history for nonexistent fighter shows error."""
    result = runner.invoke(app, ["export", "history", "nonexistent_xyz"])
    assert result.exit_code != 0


# ── Edge Cases ───────────────────────────────────────────────────────────────


def test_csv_none_handling(patch_export_session):
    """Verify None values appear as empty strings in CSV, not 'None'."""
    result = runner.invoke(app, ["export", "fighters", "--format", "csv"])
    assert result.exit_code == 0
    assert "None" not in result.stdout  # Should be empty string, not literal "None"
