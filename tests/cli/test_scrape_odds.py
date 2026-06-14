"""CLI tests for ``ufc scrape odds``.

Mocks both ``BFOScraper`` (our in-house Phase 15 scraper, replacing the
broken Python-3.14 incompatible ``ufcscraper.BestFightOddsScraper``) and
``BFOOddsIngester`` so tests run without network and without touching
the DB; DB-level integration is already covered in
``tests/scraper/test_bestfightodds_integration.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ufc_prediction.cli.main import app

runner = CliRunner()


@dataclass
class _FakeSummary:
    """Drop-in replacement for IngestSummary in unit tests."""

    bfo_fights_scanned: int = 0
    fights_matched: int = 0
    fighters_matched: int = 0
    fighters_unmatched: int = 0
    rows_upserted: int = 0
    unmatched_bfo_names: list[str] = field(default_factory=list)
    coverage_pct: float = 0.0


@dataclass
class _FakeScrapeSummary:
    """Drop-in replacement for ScrapeSummary in unit tests.

    Mirrors the shape of ``ufc_prediction.scraper.bfo_scraper.ScrapeSummary``
    so the CLI's ``console.print`` formatting reads attributes without an
    AttributeError.
    """

    fighters_processed: int = 0
    fighters_matched: int = 0
    fighters_unmatched: int = 0
    fights_emitted: int = 0
    parse_errors: int = 0
    captcha_hits: int = 0
    unmatched_db_names: list[str] = field(default_factory=list)


def _make_ingester_returning(summary: _FakeSummary) -> MagicMock:
    """Build a MagicMock BFOOddsIngester class whose .ingest_all() returns
    ``summary``. Both the class and its instance are MagicMocks so the
    CLI's ``BFOOddsIngester(session, data_folder=...)`` call and
    subsequent ``.ingest_all()`` both succeed."""
    mock_class = MagicMock()
    instance = MagicMock()
    instance.ingest_all.return_value = summary
    mock_class.return_value = instance
    return mock_class


def _make_bfo_scraper_returning(summary: _FakeScrapeSummary) -> MagicMock:
    """Build a MagicMock BFOScraper class whose .scrape_all() returns
    ``summary``. Both the class and its instance are MagicMocks so the
    CLI's ``BFOScraper(client=..., session=..., data_folder=..., workers=4)``
    call and subsequent ``.scrape_all(min_date=...)`` both succeed."""
    mock_class = MagicMock()
    instance = MagicMock()
    instance.scrape_all.return_value = summary
    mock_class.return_value = instance
    return mock_class


class TestScrapeOddsCommandDiscovery:
    """Typer registers the command and surfaces the expected options."""

    def test_scrape_odds_command_exists(self) -> None:
        result = runner.invoke(app, ["scrape", "odds", "--help"])
        assert result.exit_code == 0
        assert "BestFightOdds" in result.output

    def test_scrape_odds_help_lists_options(self) -> None:
        result = runner.invoke(app, ["scrape", "odds", "--help"])
        assert result.exit_code == 0
        # Typer wraps long option names; allow whitespace/newline between.
        for flag in ("--delay", "--n-sessions", "--min-date",
                     "--data-folder", "--skip-scrape"):
            assert flag in result.output, f"missing flag {flag} in --help"


class TestScrapeOddsSkipScrape:
    """--skip-scrape path: must NOT instantiate BFOScraper."""

    @patch("ufc_prediction.cli.main.SessionLocal")
    def test_skip_scrape_calls_ingester_only(
        self, mock_session_local: MagicMock
    ) -> None:
        mock_session_local.return_value = MagicMock()
        summary = _FakeSummary(
            bfo_fights_scanned=3,
            fights_matched=3,
            fighters_matched=6,
            rows_upserted=6,
            coverage_pct=1.0,
        )
        ingester_cls = _make_ingester_returning(summary)
        scraper_cls = _make_bfo_scraper_returning(_FakeScrapeSummary())

        with patch(
            "ufc_prediction.scraper.bfo_ingest.BFOOddsIngester", ingester_cls
        ), patch(
            "ufc_prediction.cli.main.BFOScraper", scraper_cls
        ), patch(
            "ufc_prediction.cli.main.ScraperClient"
        ):
            result = runner.invoke(
                app,
                [
                    "scrape", "odds",
                    "--skip-scrape",
                    "--data-folder", "tests/scraper/fixtures",
                ],
            )

        assert result.exit_code == 0, result.output
        ingester_cls.return_value.ingest_all.assert_called_once()
        scraper_cls.assert_not_called()


class TestScrapeOddsDefaultInvokesScraper:
    """Without --skip-scrape, BFOScraper IS instantiated +
    .scrape_all() IS called. Mocked so no network."""

    @patch("ufc_prediction.cli.main.SessionLocal")
    def test_scrape_odds_invokes_bfo_scraper_by_default(
        self, mock_session_local: MagicMock
    ) -> None:
        mock_session_local.return_value = MagicMock()
        summary = _FakeSummary()
        ingester_cls = _make_ingester_returning(summary)
        scrape_summary = _FakeScrapeSummary(
            fighters_processed=10,
            fighters_matched=8,
            fighters_unmatched=2,
            fights_emitted=42,
        )
        scraper_cls = _make_bfo_scraper_returning(scrape_summary)
        client_cls = MagicMock()
        client_cls.return_value = MagicMock()

        with patch(
            "ufc_prediction.scraper.bfo_ingest.BFOOddsIngester", ingester_cls
        ), patch(
            "ufc_prediction.cli.main.BFOScraper", scraper_cls
        ), patch(
            "ufc_prediction.cli.main.ScraperClient", client_cls
        ):
            result = runner.invoke(
                app,
                ["scrape", "odds", "--data-folder", "tests/scraper/fixtures"],
            )

        assert result.exit_code == 0, result.output
        scraper_cls.assert_called_once()
        scraper_cls.return_value.scrape_all.assert_called_once()
        # ScraperClient was constructed
        client_cls.assert_called_once()
        # client.close() must be called for cleanup
        client_cls.return_value.close.assert_called_once()

    @patch("ufc_prediction.cli.main.SessionLocal")
    def test_scrape_odds_passes_min_date_to_scraper(
        self, mock_session_local: MagicMock
    ) -> None:
        import datetime as _dt

        mock_session_local.return_value = MagicMock()
        summary = _FakeSummary()
        ingester_cls = _make_ingester_returning(summary)
        scraper_cls = _make_bfo_scraper_returning(_FakeScrapeSummary())

        with patch(
            "ufc_prediction.scraper.bfo_ingest.BFOOddsIngester", ingester_cls
        ), patch(
            "ufc_prediction.cli.main.BFOScraper", scraper_cls
        ), patch(
            "ufc_prediction.cli.main.ScraperClient"
        ):
            result = runner.invoke(
                app,
                [
                    "scrape", "odds",
                    "--min-date", "2010-01-01",
                    "--data-folder", "tests/scraper/fixtures",
                ],
            )

        assert result.exit_code == 0, result.output
        # min_date is now passed to scrape_all (not to the constructor)
        _, scrape_call_kwargs = scraper_cls.return_value.scrape_all.call_args
        assert scrape_call_kwargs.get("min_date") == _dt.date(2010, 1, 1)


class TestScrapeOddsOutput:
    """Summary table + coverage warnings render correctly."""

    @patch("ufc_prediction.cli.main.SessionLocal")
    def test_scrape_odds_prints_summary_table(
        self, mock_session_local: MagicMock
    ) -> None:
        mock_session_local.return_value = MagicMock()
        summary = _FakeSummary(
            bfo_fights_scanned=3,
            fights_matched=3,
            fighters_matched=6,
            rows_upserted=6,
            coverage_pct=1.0,
        )
        ingester_cls = _make_ingester_returning(summary)

        with patch(
            "ufc_prediction.scraper.bfo_ingest.BFOOddsIngester", ingester_cls
        ):
            result = runner.invoke(
                app,
                [
                    "scrape", "odds",
                    "--skip-scrape",
                    "--data-folder", "tests/scraper/fixtures",
                ],
            )

        assert result.exit_code == 0, result.output
        out = result.output
        # Rich may wrap labels across lines, so fuzzy-check substrings.
        assert "BFO fights scanned" in out or "BFO fights" in out
        assert "Fights matched" in out
        assert "Fighters matched" in out
        assert "Fighters unmatched" in out
        assert "Rows upserted" in out
        assert "Coverage" in out

    @patch("ufc_prediction.cli.main.SessionLocal")
    def test_scrape_odds_reports_coverage_warning_below_60pct(
        self, mock_session_local: MagicMock
    ) -> None:
        mock_session_local.return_value = MagicMock()
        summary = _FakeSummary(
            bfo_fights_scanned=10,
            fights_matched=4,
            fighters_matched=8,
            rows_upserted=8,
            coverage_pct=0.45,
        )
        ingester_cls = _make_ingester_returning(summary)

        with patch(
            "ufc_prediction.scraper.bfo_ingest.BFOOddsIngester", ingester_cls
        ):
            result = runner.invoke(
                app,
                [
                    "scrape", "odds",
                    "--skip-scrape",
                    "--data-folder", "tests/scraper/fixtures",
                ],
            )

        assert result.exit_code == 0, result.output
        # Pitfall 7 warning: mention coverage <60%.
        assert "coverage below 60" in result.output.lower()


class TestScrapeOddsPathSecurity:
    """T-13-02-03: path-traversal mitigation rejects untrusted prefixes."""

    def test_scrape_odds_rejects_unsafe_data_folder(self) -> None:
        # /etc is under the forbidden prefix list.
        result = runner.invoke(
            app, ["scrape", "odds", "--skip-scrape", "--data-folder", "/etc"]
        )
        assert result.exit_code != 0, result.output
        # Typer may print the message to stdout via rich; check combined.
        assert "unsafe" in result.output.lower() or result.exit_code == 1
