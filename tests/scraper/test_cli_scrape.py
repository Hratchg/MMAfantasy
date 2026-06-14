"""CLI integration tests for scrape commands.

Tests that 'ufc scrape all' and 'ufc scrape latest' commands exist and
invoke the orchestrator correctly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ufc_prediction.cli.main import app
from ufc_prediction.data.schemas import IngestResult

runner = CliRunner()


class TestScrapeCommandsExist:
    """Verify scrape sub-commands are registered."""

    def test_scrape_all_command_exists(self) -> None:
        result = runner.invoke(app, ["scrape", "all", "--help"])
        assert result.exit_code == 0
        assert "full historical scrape" in result.output.lower()

    def test_scrape_latest_command_exists(self) -> None:
        result = runner.invoke(app, ["scrape", "latest", "--help"])
        assert result.exit_code == 0
        assert "incremental" in result.output.lower()


class TestScrapeAllCommand:
    """Tests for 'ufc scrape all' command."""

    @patch("ufc_prediction.cli.main.SessionLocal")
    @patch("ufc_prediction.scraper.client.ScraperClient")
    @patch("ufc_prediction.cli.main.scrape_all_events")
    def test_scrape_all_calls_orchestrator(
        self,
        mock_scrape: MagicMock,
        mock_client_cls: MagicMock,
        mock_session_local: MagicMock,
    ) -> None:
        mock_scrape.return_value = IngestResult(accepted=5, updated=2)
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=mock_client,
        )
        mock_client_cls.return_value.__exit__ = MagicMock(
            return_value=False,
        )

        result = runner.invoke(app, ["scrape", "all"])

        assert result.exit_code == 0
        assert "5" in result.output
        mock_scrape.assert_called_once()

    @patch("ufc_prediction.cli.main.SessionLocal")
    @patch("ufc_prediction.scraper.client.ScraperClient")
    @patch("ufc_prediction.cli.main.scrape_all_events")
    def test_scrape_all_with_delay_option(
        self,
        mock_scrape: MagicMock,
        mock_client_cls: MagicMock,
        mock_session_local: MagicMock,
    ) -> None:
        mock_scrape.return_value = IngestResult(accepted=0)
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=mock_client,
        )
        mock_client_cls.return_value.__exit__ = MagicMock(
            return_value=False,
        )

        result = runner.invoke(app, ["scrape", "all", "--delay", "2.0"])

        assert result.exit_code == 0
        # Verify ScraperClient was called with the custom delay (workers
        # default 4 is plumbed through from the scrape_all command).
        mock_client_cls.assert_called_once_with(delay=2.0, workers=4)


class TestScrapeLatestCommand:
    """Tests for 'ufc scrape latest' command."""

    @patch("ufc_prediction.cli.main.SessionLocal")
    @patch("ufc_prediction.scraper.client.ScraperClient")
    @patch("ufc_prediction.cli.main.scrape_latest_events")
    def test_scrape_latest_calls_orchestrator(
        self,
        mock_scrape: MagicMock,
        mock_client_cls: MagicMock,
        mock_session_local: MagicMock,
    ) -> None:
        mock_scrape.return_value = IngestResult(accepted=0)
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=mock_client,
        )
        mock_client_cls.return_value.__exit__ = MagicMock(
            return_value=False,
        )

        result = runner.invoke(app, ["scrape", "latest"])

        assert result.exit_code == 0
        assert "up to date" in result.output.lower()
        mock_scrape.assert_called_once()


# ── --workers CLI option (quick task 260422-qye) ────────────────────────────


class TestScrapeWorkersOption:
    """Tests that --workers plumbs through to ScraperClient and the orchestrator."""

    @patch("ufc_prediction.cli.main.SessionLocal")
    @patch("ufc_prediction.scraper.client.ScraperClient")
    @patch("ufc_prediction.cli.main.scrape_all_events")
    def test_scrape_all_defaults_to_workers_4(
        self,
        mock_scrape: MagicMock,
        mock_client_cls: MagicMock,
        mock_session_local: MagicMock,
    ) -> None:
        mock_scrape.return_value = IngestResult(accepted=0)
        mock_session_local.return_value = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = runner.invoke(app, ["scrape", "all"])

        assert result.exit_code == 0
        _, kwargs = mock_client_cls.call_args
        assert kwargs.get("workers") == 4

    @patch("ufc_prediction.cli.main.SessionLocal")
    @patch("ufc_prediction.scraper.client.ScraperClient")
    @patch("ufc_prediction.cli.main.scrape_all_events")
    def test_scrape_all_accepts_workers_override(
        self,
        mock_scrape: MagicMock,
        mock_client_cls: MagicMock,
        mock_session_local: MagicMock,
    ) -> None:
        mock_scrape.return_value = IngestResult(accepted=0)
        mock_session_local.return_value = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = runner.invoke(app, ["scrape", "all", "--workers", "2"])

        assert result.exit_code == 0
        _, kwargs = mock_client_cls.call_args
        assert kwargs.get("workers") == 2

    def test_scrape_all_rejects_workers_zero(self) -> None:
        # Do NOT mock ScraperClient — we want the real validation to fire
        # (ScraperClient(workers=0) raises ValueError). The CLI's outer
        # except-Exception handler converts that to a non-zero exit code.
        result = runner.invoke(app, ["scrape", "all", "--workers", "0"])
        assert result.exit_code != 0

    @patch("ufc_prediction.cli.main.SessionLocal")
    @patch("ufc_prediction.scraper.client.ScraperClient")
    @patch("ufc_prediction.cli.main.scrape_latest_events")
    def test_scrape_latest_defaults_to_workers_4(
        self,
        mock_scrape: MagicMock,
        mock_client_cls: MagicMock,
        mock_session_local: MagicMock,
    ) -> None:
        mock_scrape.return_value = IngestResult(accepted=0)
        mock_session_local.return_value = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = runner.invoke(app, ["scrape", "latest"])

        assert result.exit_code == 0
        _, kwargs = mock_client_cls.call_args
        assert kwargs.get("workers") == 4

    @patch("ufc_prediction.cli.main.SessionLocal")
    @patch("ufc_prediction.scraper.client.ScraperClient")
    @patch("ufc_prediction.cli.main.scrape_latest_events")
    def test_scrape_latest_accepts_workers_override(
        self,
        mock_scrape: MagicMock,
        mock_client_cls: MagicMock,
        mock_session_local: MagicMock,
    ) -> None:
        mock_scrape.return_value = IngestResult(accepted=0)
        mock_session_local.return_value = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = runner.invoke(app, ["scrape", "latest", "--workers", "2"])

        assert result.exit_code == 0
        _, kwargs = mock_client_cls.call_args
        assert kwargs.get("workers") == 2


# ── `ufc scrape sherdog` --workers / --since-year (quick task 260423-agz) ──


class TestScrapeSherdogCommand:
    """Tests for `ufc scrape sherdog` --workers / --since-year wiring."""

    @patch("ufc_prediction.cli.main.SessionLocal")
    @patch("ufc_prediction.scraper.client.ScraperClient")
    @patch("ufc_prediction.cli.main.SherdogScraper")
    def test_sherdog_defaults_workers_2_since_none(
        self,
        mock_scraper_cls: MagicMock,
        mock_client_cls: MagicMock,
        mock_session_local: MagicMock,
    ) -> None:
        mock_session_local.return_value = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_scraper = MagicMock()
        mock_scraper.scrape_all_fighters.return_value = {
            "matched": 0, "unmatched": 0, "skipped": 0, "errors": 0,
            "unmatched_names": [],
        }
        mock_scraper_cls.return_value = mock_scraper

        result = runner.invoke(app, ["scrape", "sherdog"])

        assert result.exit_code == 0
        # ScraperClient built with workers=2
        _, client_kwargs = mock_client_cls.call_args
        assert client_kwargs.get("workers") == 2
        # SherdogScraper built with workers=2
        _, scraper_kwargs = mock_scraper_cls.call_args
        assert scraper_kwargs.get("workers") == 2
        # since_year=None passed through
        _, call_kwargs = mock_scraper.scrape_all_fighters.call_args
        assert call_kwargs.get("since_year") is None

    @patch("ufc_prediction.cli.main.SessionLocal")
    @patch("ufc_prediction.scraper.client.ScraperClient")
    @patch("ufc_prediction.cli.main.SherdogScraper")
    def test_sherdog_accepts_workers_override(
        self,
        mock_scraper_cls: MagicMock,
        mock_client_cls: MagicMock,
        mock_session_local: MagicMock,
    ) -> None:
        mock_session_local.return_value = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_scraper = MagicMock()
        mock_scraper.scrape_all_fighters.return_value = {
            "matched": 0, "unmatched": 0, "skipped": 0, "errors": 0,
            "unmatched_names": [],
        }
        mock_scraper_cls.return_value = mock_scraper

        result = runner.invoke(app, ["scrape", "sherdog", "--workers", "4"])

        assert result.exit_code == 0
        _, client_kwargs = mock_client_cls.call_args
        assert client_kwargs.get("workers") == 4
        _, scraper_kwargs = mock_scraper_cls.call_args
        assert scraper_kwargs.get("workers") == 4

    @patch("ufc_prediction.cli.main.SessionLocal")
    @patch("ufc_prediction.scraper.client.ScraperClient")
    @patch("ufc_prediction.cli.main.SherdogScraper")
    def test_sherdog_accepts_since_year(
        self,
        mock_scraper_cls: MagicMock,
        mock_client_cls: MagicMock,
        mock_session_local: MagicMock,
    ) -> None:
        mock_session_local.return_value = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_scraper = MagicMock()
        mock_scraper.scrape_all_fighters.return_value = {
            "matched": 0, "unmatched": 0, "skipped": 0, "errors": 0,
            "unmatched_names": [],
        }
        mock_scraper_cls.return_value = mock_scraper

        result = runner.invoke(
            app, ["scrape", "sherdog", "--since-year", "2015"]
        )

        assert result.exit_code == 0
        _, call_kwargs = mock_scraper.scrape_all_fighters.call_args
        assert call_kwargs.get("since_year") == 2015

    def test_sherdog_rejects_workers_zero(self) -> None:
        # Do NOT mock ScraperClient — let the real validation fire
        # (ScraperClient(workers=0) raises ValueError). The CLI's outer
        # except-Exception handler converts that to a non-zero exit code.
        result = runner.invoke(app, ["scrape", "sherdog", "--workers", "0"])
        assert result.exit_code != 0

    def test_sherdog_help_lists_new_options(self) -> None:
        """--help output documents both --workers and --since-year."""
        result = runner.invoke(app, ["scrape", "sherdog", "--help"])
        assert result.exit_code == 0
        assert "--workers" in result.output
        assert "--since-year" in result.output
