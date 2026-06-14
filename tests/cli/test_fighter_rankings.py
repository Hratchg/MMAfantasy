"""Integration tests for the `ufc fighter rankings` CLI command."""

from __future__ import annotations

from typer.testing import CliRunner

from ufc_prediction.cli.main import app

runner = CliRunner()


class TestFighterRankings:
    def test_rankings_default(self, patch_session):
        result = runner.invoke(app, ["fighter", "rankings", "Lightweight"])
        assert result.exit_code == 0
        assert "Lightweight Rankings" in result.output
        assert "Khabib Nurmagomedov" in result.output

    def test_rankings_custom_top(self, patch_session):
        result = runner.invoke(
            app, ["fighter", "rankings", "Lightweight", "--top", "2"]
        )
        assert result.exit_code == 0
        # Should show exactly 2 rows (Khabib and Dustin, not Conor)
        assert "Khabib Nurmagomedov" in result.output
        # Count occurrences of table row markers to verify limiting
        lines = [
            line for line in result.output.splitlines()
            if "Nurmagomedov" in line or "Poirier" in line or "McGregor" in line
        ]
        assert len(lines) <= 2

    def test_rankings_partial_match_ambiguous(self, patch_session):
        result = runner.invoke(app, ["fighter", "rankings", "light"])
        assert result.exit_code == 0
        assert "Ambiguous" in result.output
        assert "Lightweight" in result.output
        assert "Light Heavyweight" in result.output

    def test_rankings_no_match(self, patch_session):
        result = runner.invoke(app, ["fighter", "rankings", "ZZZZZ"])
        assert result.exit_code == 0
        assert "No weight class matching" in result.output

    def test_rankings_ordered_by_elo(self, patch_session):
        """Verify that the highest-Elo fighter appears first in output."""
        result = runner.invoke(app, ["fighter", "rankings", "Lightweight"])
        assert result.exit_code == 0
        output = result.output
        # Khabib (1555) should appear before Dustin (1498) and Conor (1448)
        khabib_pos = output.index("Khabib Nurmagomedov")
        dustin_pos = output.index("Dustin Poirier")
        conor_pos = output.index("Conor McGregor")
        assert khabib_pos < dustin_pos < conor_pos

    def test_rankings_no_duplicate_fighters(self, patch_session):
        """DEDUP-03: each fighter appears at most once per division output."""
        result = runner.invoke(app, ["fighter", "rankings", "Lightweight"])
        assert result.exit_code == 0
        # Existing seed has Khabib, Conor, Dustin in Lightweight via Elo
        # snapshots. Each name must appear exactly once in the rendered
        # rankings table.
        assert result.output.count("Khabib Nurmagomedov") == 1
        assert result.output.count("Conor McGregor") == 1
        assert result.output.count("Dustin Poirier") == 1
        # DEDUP-03 regression: duplicate_ufc and duplicate_kaggle both have
        # EloSnapshot rows in Lightweight. The dedup path in
        # get_division_rankings must collapse them to a single entry.
        assert result.output.count("Duplicate Bob") == 1
