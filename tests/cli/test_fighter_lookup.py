"""Integration tests for the `ufc fighter lookup` CLI command."""

from __future__ import annotations

from typer.testing import CliRunner

from ufc_prediction.cli.main import app

runner = CliRunner()


class TestFighterLookup:
    def test_lookup_single_match(self, patch_session):
        result = runner.invoke(app, ["fighter", "lookup", "Khabib"])
        assert result.exit_code == 0
        assert "Khabib Nurmagomedov" in result.output
        assert "Elo Rating" in result.output
        assert "2W - 0L" in result.output

    def test_lookup_multiple_matches(self, patch_session):
        result = runner.invoke(app, ["fighter", "lookup", "Silva"])
        assert result.exit_code == 0
        assert "Multiple fighters matching" in result.output
        assert "Anderson Silva" in result.output
        assert "Antonio Silva" in result.output

    def test_lookup_no_match(self, patch_session):
        result = runner.invoke(app, ["fighter", "lookup", "ZZZZNOTANAME"])
        assert result.exit_code == 0
        assert "No fighters found matching" in result.output

    def test_lookup_by_alias(self, patch_session):
        result = runner.invoke(app, ["fighter", "lookup", "The Eagle"])
        assert result.exit_code == 0
        assert "Khabib Nurmagomedov" in result.output

    def test_lookup_case_insensitive(self, patch_session):
        result = runner.invoke(app, ["fighter", "lookup", "khabib"])
        assert result.exit_code == 0
        assert "Khabib Nurmagomedov" in result.output

    def test_lookup_dedups_multi_source_fighter(self, patch_session):
        """DEDUP-03: same name with ufcstats + kaggle source rows collapses to one entry."""
        result = runner.invoke(app, ["fighter", "lookup", "Duplicate Bob"])
        assert result.exit_code == 0
        # Multi-match table must NOT appear when the only "duplicates" are
        # the same person across sources.
        assert "Multiple fighters matching" not in result.output
        # The canonical fighter's name should appear (the detail view header).
        assert "Duplicate Bob" in result.output
        # Specifically, name should appear at most twice (header + maybe a
        # division-table row); never 4+ times which would indicate two
        # detail tables were rendered.
        assert result.output.count("Duplicate Bob") <= 3
