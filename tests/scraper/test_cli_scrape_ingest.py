"""Phase 28 INGEST-V23 CLI tests — `ufc scrape referees` + `ufc scrape venues`.

Covers:
- --help works for both subcommands
- --dry-run prints ETA + counts without invoking the slim driver / DB writes
- Missing --confirm + missing --dry-run exits 1 (CONTEXT D-08 safety gate)
- --confirm invokes the underlying driver / writes FKs idempotently
- Venue path emits 28-UNMATCHED-VENUES.md when there are unresolvable strings

Uses Typer's CliRunner pattern (consistent with tests/scraper/test_cli_scrape.py).
Mocks the slim referee driver to keep tests hermetic; the venue path uses the
real session fixture from tests/conftest.py since it has no HTTP and can
populate Event + Venue rows in-memory.
"""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from ufc_prediction.cli.main import app
from ufc_prediction.models.event import Event
from ufc_prediction.models.venue import Venue

runner = CliRunner()


# ── Referee CLI tests (mocked slim driver — hermetic, no DB) ──────────────


class TestScrapeRefereesCLIHelp:
    def test_referees_help_returns_zero(self) -> None:
        result = runner.invoke(app, ["scrape", "referees", "--help"])
        assert result.exit_code == 0
        assert "--dry-run" in result.output
        assert "--confirm" in result.output


class TestScrapeRefereesCLI:
    """Drives `ufc scrape referees`. Patches the slim driver so no HTTP/DB
    side effects occur."""

    def test_dry_run_invokes_driver_with_dry_run_true(self) -> None:
        with patch(
            "ufc_prediction.cli.main._scrape_referees_full.main",
            return_value=0,
        ) as mock_main:
            result = runner.invoke(app, ["scrape", "referees", "--dry-run"])
            assert result.exit_code == 0, result.output
            mock_main.assert_called_once()
            kwargs = mock_main.call_args.kwargs
            assert kwargs["dry_run"] is True
            assert kwargs["confirm"] is False

    def test_no_confirm_no_dry_run_returns_exit_1(self) -> None:
        """CONTEXT D-08 safety gate — refuse default-confirm."""
        with patch(
            "ufc_prediction.cli.main._scrape_referees_full.main",
            return_value=1,
        ) as mock_main:
            result = runner.invoke(app, ["scrape", "referees"])
            assert result.exit_code == 1
            mock_main.assert_called_once()
            kwargs = mock_main.call_args.kwargs
            assert kwargs["dry_run"] is False
            assert kwargs["confirm"] is False

    def test_confirm_invokes_driver_with_confirm_true(self) -> None:
        with patch(
            "ufc_prediction.cli.main._scrape_referees_full.main",
            return_value=0,
        ) as mock_main:
            result = runner.invoke(app, ["scrape", "referees", "--confirm"])
            assert result.exit_code == 0, result.output
            kwargs = mock_main.call_args.kwargs
            assert kwargs["confirm"] is True
            assert kwargs["dry_run"] is False

    def test_custom_delay_and_workers_flow_through(self) -> None:
        with patch(
            "ufc_prediction.cli.main._scrape_referees_full.main",
            return_value=0,
        ) as mock_main:
            result = runner.invoke(
                app,
                ["scrape", "referees", "--delay", "2.0", "--workers", "8", "--dry-run"],
            )
            assert result.exit_code == 0, result.output
            kwargs = mock_main.call_args.kwargs
            assert kwargs["delay"] == 2.0
            assert kwargs["workers"] == 8


# ── Referee slim-driver dry-run unit test (no DB; pure stdout shape) ──────


class TestScrapeRefereesDriverDryRun:
    """Direct unit tests on scripts.scrape_referees_full._print_dry_run_summary
    — the stdout shape that the CLI's dry-run path produces. No DB, no HTTP."""

    def test_dry_run_summary_prints_eta_keys(self, capsys: pytest.CaptureFixture[str]) -> None:
        from scripts.scrape_referees_full import _print_dry_run_summary

        _print_dry_run_summary(
            events_to_process=1872,
            cache_hits_expected=500,
            delay=1.2,
            workers=4,
        )
        captured = capsys.readouterr().out
        assert "Events to process: 1872" in captured
        assert "Cache hits expected: 500" in captured
        assert "ETA:" in captured

    def test_dry_run_zero_events_yields_zero_eta(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from scripts.scrape_referees_full import _print_dry_run_summary

        _print_dry_run_summary(
            events_to_process=0,
            cache_hits_expected=0,
            delay=1.2,
            workers=4,
        )
        captured = capsys.readouterr().out
        assert "ETA: 0.0 minutes" in captured


# ── Venue CLI tests (real DB via testcontainer; no HTTP) ──────────────────


@pytest.fixture
def venues_csv_tmp(tmp_path: Path) -> Path:
    """Write a tiny venues.csv with Vegas + Tokyo for the venue CLI to load."""
    p = tmp_path / "venues.csv"
    with p.open("w", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "venue_id",
                "name",
                "city",
                "state",
                "country",
                "lat",
                "lon",
                "timezone_iana",
                "n_events",
                "geocode_source",
            ]
        )
        w.writerow(
            [
                "9001",
                "Las Vegas, Nevada, USA",
                "Las Vegas",
                "Nevada",
                "USA",
                "36.1674",
                "-115.1484",
                "America/Los_Angeles",
                "1",
                "test",
            ]
        )
        w.writerow(
            [
                "9002",
                "Tokyo, Japan",
                "Tokyo",
                "",
                "Japan",
                "35.6762",
                "139.6503",
                "Asia/Tokyo",
                "1",
                "test",
            ]
        )
    return p


@pytest.fixture
def seed_venues_and_events(session: Session) -> dict[str, int]:
    """Pre-seed Venue rows (matching venues_csv_tmp) + a few Event rows with
    NULL venue_id. Returns id-mapping for assertions."""
    session.add_all(
        [
            Venue(
                id=9001,
                name="Las Vegas, Nevada, USA",
                city="Las Vegas",
                state="Nevada",
                country="USA",
                lat=36.1674,
                lon=-115.1484,
                timezone_iana="America/Los_Angeles",
            ),
            Venue(
                id=9002,
                name="Tokyo, Japan",
                city="Tokyo",
                state=None,
                country="Japan",
                lat=35.6762,
                lon=139.6503,
                timezone_iana="Asia/Tokyo",
            ),
        ]
    )
    session.flush()
    e1 = Event(
        name="UFC Test Vegas",
        date=date(2026, 5, 1),
        location="Las Vegas, Nevada, USA",
        source="ufcstats",
        source_url="http://example.com/event/test-vegas",
    )
    e2 = Event(
        name="UFC Test Tokyo Variant",
        date=date(2026, 5, 2),
        # Word-order swap of "Tokyo, Japan" — token_sort_ratio ~92, above
        # the 85 fuzzy threshold; fuzz.ratio would score ~25 (below).
        location="Japan, Tokyo",
        source="ufcstats",
        source_url="http://example.com/event/test-tokyo",
    )
    e3 = Event(
        name="UFC Test Mars",
        date=date(2026, 5, 3),
        location="Mars Crater, Olympus Mons",  # unresolvable
        source="ufcstats",
        source_url="http://example.com/event/test-mars",
    )
    session.add_all([e1, e2, e3])
    session.flush()
    return {"vegas_event": e1.id, "tokyo_event": e2.id, "mars_event": e3.id}


class TestScrapeVenuesCLIHelp:
    def test_venues_help_returns_zero(self) -> None:
        result = runner.invoke(app, ["scrape", "venues", "--help"])
        assert result.exit_code == 0
        assert "--dry-run" in result.output
        assert "--confirm" in result.output


class TestScrapeVenuesCLIDryRun:
    """Dry-run prints counts; no DB writes."""

    def test_dry_run_prints_counts(
        self,
        session: Session,
        seed_venues_and_events: dict[str, int],
        venues_csv_tmp: Path,
        tmp_path: Path,
    ) -> None:
        # Wire SessionLocal to the testcontainer-bound session so the CLI
        # picks up the seeded rows.
        with patch("ufc_prediction.cli.main.SessionLocal", return_value=session):
            result = runner.invoke(
                app,
                [
                    "scrape", "venues",
                    "--dry-run",
                    "--venues-csv", str(venues_csv_tmp),
                    "--unmatched-output", str(tmp_path / "unmatched.md"),
                ],
            )
        assert result.exit_code == 0, result.output
        assert "Events to process:" in result.output
        assert "Unresolvable distinct strings:" in result.output
        # No DB write occurred — events should still have NULL venue_id.
        for ev_id in seed_venues_and_events.values():
            ev = session.get(Event, ev_id)
            assert ev.venue_id is None
        # No unmatched file written on dry-run.
        assert not (tmp_path / "unmatched.md").exists()


class TestScrapeVenuesCLISafetyGate:
    def test_no_confirm_no_dry_run_exits_1(
        self,
        session: Session,
        seed_venues_and_events: dict[str, int],
        venues_csv_tmp: Path,
        tmp_path: Path,
    ) -> None:
        with patch("ufc_prediction.cli.main.SessionLocal", return_value=session):
            result = runner.invoke(
                app,
                [
                    "scrape", "venues",
                    "--venues-csv", str(venues_csv_tmp),
                    "--unmatched-output", str(tmp_path / "unmatched.md"),
                ],
            )
        assert result.exit_code == 1
        # No DB writes happened (safety gate prevented).
        for ev_id in seed_venues_and_events.values():
            ev = session.get(Event, ev_id)
            assert ev.venue_id is None


class TestScrapeVenuesCLIConfirm:
    """End-to-end confirm path with a real DB session."""

    def test_confirm_assigns_venue_ids_idempotently(
        self,
        session: Session,
        seed_venues_and_events: dict[str, int],
        venues_csv_tmp: Path,
        tmp_path: Path,
    ) -> None:
        ids = seed_venues_and_events
        # First run — should assign Vegas (exact), Tokyo (fuzzy), Mars (unmatched).
        with patch("ufc_prediction.cli.main.SessionLocal", return_value=session):
            result1 = runner.invoke(
                app,
                [
                    "scrape", "venues",
                    "--confirm",
                    "--venues-csv", str(venues_csv_tmp),
                    "--unmatched-output", str(tmp_path / "unmatched.md"),
                ],
            )
        assert result1.exit_code == 0, result1.output
        session.expire_all()  # re-read from DB
        vegas_ev = session.get(Event, ids["vegas_event"])
        tokyo_ev = session.get(Event, ids["tokyo_event"])
        mars_ev = session.get(Event, ids["mars_event"])
        assert vegas_ev.venue_id == 9001  # exact
        assert tokyo_ev.venue_id == 9002  # fuzzy
        assert mars_ev.venue_id is None    # unmatched

        # Second run — should be a no-op for set FKs (CR-01 idempotency).
        # Capture pre-state.
        pre = {
            "vegas": vegas_ev.venue_id,
            "tokyo": tokyo_ev.venue_id,
        }
        with patch("ufc_prediction.cli.main.SessionLocal", return_value=session):
            result2 = runner.invoke(
                app,
                [
                    "scrape", "venues",
                    "--confirm",
                    "--venues-csv", str(venues_csv_tmp),
                    "--unmatched-output", str(tmp_path / "unmatched.md"),
                ],
            )
        assert result2.exit_code == 0, result2.output
        session.expire_all()
        # FKs preserved verbatim.
        assert session.get(Event, ids["vegas_event"]).venue_id == pre["vegas"]
        assert session.get(Event, ids["tokyo_event"]).venue_id == pre["tokyo"]

    def test_unmatched_emits_unmatched_artifact(
        self,
        session: Session,
        seed_venues_and_events: dict[str, int],
        venues_csv_tmp: Path,
        tmp_path: Path,
    ) -> None:
        out_path = tmp_path / "28-UNMATCHED-VENUES.md"
        with patch("ufc_prediction.cli.main.SessionLocal", return_value=session):
            result = runner.invoke(
                app,
                [
                    "scrape", "venues",
                    "--confirm",
                    "--venues-csv", str(venues_csv_tmp),
                    "--unmatched-output", str(out_path),
                ],
            )
        assert result.exit_code == 0, result.output
        assert out_path.exists(), "unmatched-venues artifact must be emitted"
        text = out_path.read_text(encoding="utf-8")
        assert "Mars Crater" in text  # the unmatched location
        assert "n_events" in text     # header row of the table
        assert "Phase 28" in text     # the report title
