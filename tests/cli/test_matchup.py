"""Integration tests for the CLI matchup command."""

from __future__ import annotations

from typer.testing import CliRunner

from ufc_prediction.cli.main import app

runner = CliRunner()


class TestMatchupFullOutput:
    """Tests for the full matchup comparison output."""

    def test_matchup_full_output(self, patch_session):
        """ufc matchup Khabib Conor resolves both names and shows all sections."""
        result = runner.invoke(app, ["matchup", "Khabib", "Conor"])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "Physical Comparison" in result.output
        assert "Elo Comparison" in result.output
        assert "Striking Matchup" in result.output
        assert "Grappling Matchup" in result.output
        assert "Style Analysis" in result.output

    def test_matchup_style_index_displayed(self, patch_session):
        """Style Analysis section shows style tags and style index values."""
        result = runner.invoke(app, ["matchup", "Khabib", "Conor"])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        # Khabib has style_tag=grappler, Conor has style_tag=striker
        assert "grappler" in result.output.lower()
        assert "striker" in result.output.lower()
        # Style index values should be present (striking_elo - grappling_elo)
        # Khabib: 1515 - 1535 = -20.0, Conor: 1525 - 1475 = 50.0
        assert "-20" in result.output
        assert "50" in result.output


class TestMatchupEdgeCases:
    """Tests for ambiguous, unknown, and same-fighter edge cases."""

    def test_matchup_ambiguous_name(self, patch_session):
        """Ambiguous name (Silva) shows multiple candidates."""
        result = runner.invoke(app, ["matchup", "Silva", "Khabib"])
        # Both Anderson Silva and Antonio Silva match "Silva"
        assert "Multiple matches" in result.output

    def test_matchup_unknown_fighter(self, patch_session):
        """Unknown fighter name shows error message."""
        result = runner.invoke(app, ["matchup", "NonExistent", "Khabib"])
        assert "No fighters found" in result.output

    def test_matchup_same_fighter(self, patch_session):
        """Both names resolving to same fighter shows error."""
        result = runner.invoke(app, ["matchup", "Khabib", "Nurmagomedov"])
        assert "Cannot compare a fighter to themselves" in result.output


class TestMatchupMissingData:
    """Tests for handling missing data gracefully."""

    def test_matchup_missing_features(self, patch_session):
        """Fighter with no ComputedFeature rows shows N/A without crashing."""
        # Dustin has no ComputedFeature rows in seed data
        result = runner.invoke(app, ["matchup", "Dustin", "Khabib"])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "N/A" in result.output
