"""Typer CLI for the UFC Fight Prediction application.

Provides commands for data ingestion from Kaggle datasets,
Elo rating computation, and backtesting.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC
from datetime import date as date_type
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ufc_prediction.cli.db import db_app
from ufc_prediction.cli.export import export_app
from ufc_prediction.cli.predict import predict_app
from ufc_prediction.data.download import ensure_dataset
from ufc_prediction.data.ingest_mdabbert import ingest_mdabbert
from ufc_prediction.data.ingest_rajeevw import ingest_rajeevw_fighters, ingest_rajeevw_fights
from ufc_prediction.data.schemas import IngestResult
from ufc_prediction.db.session import SessionLocal
from ufc_prediction.elo.backtesting import (
    evaluate_domain_brier,
    evaluate_ewma_brier,
    generate_domain_attribution_grid,
    generate_ewma_grid,
    generate_grid_configs,
    generate_inactivity_grid,
    generate_mov_grid,
    generate_transfer_grid,
    load_backtest_results,
    run_backtest,
    run_joint_optimization,
    save_backtest_results,
)
from ufc_prediction.elo.config import EloConfig
from ufc_prediction.elo.domain import DomainEloComputer
from ufc_prediction.elo.engine import EloEngine
from ufc_prediction.elo.seed import load_seeds

# DEBUT-V25-03 (Phase 43): canonical Sherdog pre-UFC records substrate path.
# compute_elo loads seeds from this CSV at engine construction. Missing-file
# fallback returns {} (load_seeds contract) so the engine degrades cleanly
# to the pre-Phase-43 flat-1500 default.
_SHERDOG_PRE_UFC_CSV = Path("data/sherdog/pre_ufc_records.csv")
from ufc_prediction.elo.fighter_queries import (
    get_division_rankings,
    get_fighter_detail,
    get_fighter_divisions,
    get_fighter_domain_elo,
    list_divisions,
    resolve_weight_class,
    search_fighters,
)
from ufc_prediction.elo.queries import (
    flush_domain_snapshots,
    flush_snapshots,
    load_fights_chronological,
    load_round_stats_by_fight,
)
from ufc_prediction.ml.predictor import _prefer_canonical
from ufc_prediction.models.fighter import Fighter
from ufc_prediction.scraper.bfo_scraper import BFOScraper
from ufc_prediction.scraper.client import ScraperClient
from ufc_prediction.scraper.ingest import scrape_all_events, scrape_latest_events
from ufc_prediction.scraper.sherdog import SherdogScraper

app = typer.Typer(name="ufc", help="UFC Fight Prediction CLI")
ingest_app = typer.Typer(name="ingest", help="Data ingestion commands")
app.add_typer(ingest_app)
elo_app = typer.Typer(name="elo", help="Elo rating computation and analysis")
app.add_typer(elo_app)
fighter_app = typer.Typer(name="fighter", help="Fighter lookup and rankings")
app.add_typer(fighter_app)
scrape_app = typer.Typer(name="scrape", help="UFCStats.com scraping commands")
app.add_typer(scrape_app)
features_app = typer.Typer(name="features", help="Feature computation commands")
app.add_typer(features_app)
matchup_app = typer.Typer(name="matchup", help="Fighter matchup comparison")
app.add_typer(matchup_app)
# Phase 56 GATE-V26-03 — gate verification + GATE-RECALIB-PERIODIC CLI.
gate_app = typer.Typer(
    name="gate",
    help="Gate verification + threshold recalibration (v2.6 GATE-V26-03).",
)
app.add_typer(gate_app)
app.add_typer(export_app)
app.add_typer(predict_app)
app.add_typer(db_app)
console = Console()

logger = logging.getLogger(__name__)


def _print_summary_table(results: dict[str, IngestResult]) -> None:
    """Print a Rich summary table for ingestion results."""
    table = Table(title="Ingestion Summary")
    table.add_column("Step", style="cyan")
    table.add_column("Accepted", style="green", justify="right")
    table.add_column("Updated", style="yellow", justify="right")
    table.add_column("Rejected", style="red", justify="right")

    total_rejected = 0
    for step_name, result in results.items():
        table.add_row(
            step_name,
            str(result.accepted),
            str(result.updated),
            str(result.rejected),
        )
        total_rejected += result.rejected

    console.print(table)

    if total_rejected > 0:
        console.print(
            f"\n[yellow]{total_rejected} row(s) rejected.[/yellow] "
            "Run with --verbose to see rejection details."
        )


def _resolve_dataset(
    dataset_slug: str,
    data_dir: Path,
    skip_download: bool,
) -> Path:
    """Resolve dataset path, handling download or local fallback.

    Returns the path to the dataset directory.
    Exits with code 1 if dataset cannot be found.
    """
    if skip_download:
        # Only check local paths
        slug_parts = dataset_slug.split("/")
        candidates = [
            data_dir / "-".join(slug_parts),
            data_dir / slug_parts[-1],
            data_dir,
        ]
        for candidate in candidates:
            if candidate.is_dir() and any(candidate.glob("*.csv")):
                return candidate
        console.print(
            f"[red]Dataset {dataset_slug} not found locally.[/red]\n"
            f"Download from https://www.kaggle.com/datasets/{dataset_slug}\n"
            f"and place CSV files in {data_dir}/{slug_parts[-1]}/"
        )
        raise SystemExit(1)

    try:
        return ensure_dataset(dataset_slug, data_dir)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc


def _run_rajeevw_pipeline(
    data_dir: Path,
    skip_download: bool,
) -> dict[str, IngestResult]:
    """Run rajeevw ingestion pipeline and return results."""
    rajeevw_path = _resolve_dataset("rajeevw/ufcdata", data_dir, skip_download)

    results: dict[str, IngestResult] = {}
    session = SessionLocal()
    try:
        console.print("[bold]Ingesting rajeevw fighters...[/bold]")
        results["Rajeevw Fighters"] = ingest_rajeevw_fighters(
            rajeevw_path / "raw_fighter_details.csv", session
        )
        console.print(f"  {results['Rajeevw Fighters'].summary()}")

        console.print("[bold]Ingesting rajeevw fights...[/bold]")
        results["Rajeevw Fights"] = ingest_rajeevw_fights(
            rajeevw_path / "raw_total_fight_data.csv", session
        )
        console.print(f"  {results['Rajeevw Fights'].summary()}")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return results


def _run_mdabbert_pipeline(
    data_dir: Path,
    skip_download: bool,
) -> dict[str, IngestResult]:
    """Run mdabbert ingestion pipeline and return results."""
    mdabbert_path = _resolve_dataset("mdabbert/ultimate-ufc-dataset", data_dir, skip_download)

    results: dict[str, IngestResult] = {}
    session = SessionLocal()
    try:
        console.print("[bold]Ingesting mdabbert supplement...[/bold]")
        results["Mdabbert Supplement"] = ingest_mdabbert(mdabbert_path / "ufc-master.csv", session)
        console.print(f"  {results['Mdabbert Supplement'].summary()}")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return results


@ingest_app.command("all")
def ingest_all(
    data_dir: Annotated[Path, typer.Option(help="Directory containing dataset files")] = Path(
        "data"
    ),
    skip_download: Annotated[
        bool, typer.Option(help="Skip kagglehub download, use local files only")
    ] = False,
) -> None:
    """Ingest both Kaggle datasets into the database.

    Runs the full ingestion pipeline:
    1. Download/locate datasets
    2. Ingest rajeevw fighters (raw_fighter_details.csv)
    3. Ingest rajeevw fights (raw_total_fight_data.csv)
    4. Ingest mdabbert supplement (ufc-master.csv)
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    all_results: dict[str, IngestResult] = {}
    all_results.update(_run_rajeevw_pipeline(data_dir, skip_download))
    all_results.update(_run_mdabbert_pipeline(data_dir, skip_download))

    _print_summary_table(all_results)


@ingest_app.command("rajeevw")
def ingest_rajeevw_cmd(
    data_dir: Annotated[Path, typer.Option(help="Directory containing dataset files")] = Path(
        "data"
    ),
    skip_download: Annotated[
        bool, typer.Option(help="Skip kagglehub download, use local files only")
    ] = False,
) -> None:
    """Ingest the rajeevw/ufcdata dataset (fighters + fights).

    Runs:
    1. Download/locate rajeevw dataset
    2. Ingest fighters from raw_fighter_details.csv
    3. Ingest fights from raw_total_fight_data.csv
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results = _run_rajeevw_pipeline(data_dir, skip_download)
    _print_summary_table(results)


@ingest_app.command("mdabbert")
def ingest_mdabbert_cmd(
    data_dir: Annotated[Path, typer.Option(help="Directory containing dataset files")] = Path(
        "data"
    ),
    skip_download: Annotated[
        bool, typer.Option(help="Skip kagglehub download, use local files only")
    ] = False,
) -> None:
    """Ingest the mdabbert/ultimate-ufc-dataset (supplement).

    Runs:
    1. Download/locate mdabbert dataset
    2. Supplement fighter profiles and add missing fights
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results = _run_mdabbert_pipeline(data_dir, skip_download)
    _print_summary_table(results)


# ── Scrape Commands ──────────────────────────────────────────────────────────


@scrape_app.command("all")
def scrape_all(
    delay: Annotated[float, typer.Option(help="Delay between requests in seconds")] = 1.5,
    after: Annotated[
        str | None, typer.Option(help="Only scrape events after this date (YYYY-MM-DD)")
    ] = None,
    workers: Annotated[
        int,
        typer.Option(
            help="Concurrent worker threads (default 4). Reduce to 1 if rate-limited (429s).",
        ),
    ] = 4,
) -> None:
    """Scrape all completed events from UFCStats.com (full historical scrape).

    Downloads event listings, fight details, and fighter profiles.
    Uses upsert logic -- safe to re-run without creating duplicates.
    Use --after YYYY-MM-DD to resume from a specific date.
    Estimated time: 2-3 hours with default 1.5s delay and workers=4.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from ufc_prediction.scraper.client import ScraperClient

    after_date = None
    if after:
        from datetime import date as dt_date

        after_date = dt_date.fromisoformat(after)
        console.print(f"[bold]Resuming UFCStats scrape after {after}...[/bold]")
    else:
        console.print("[bold]Starting full UFCStats scrape...[/bold]")
    console.print(f"Request delay: {delay}s")
    console.print(f"Workers: {workers}")

    session = SessionLocal()
    try:
        with ScraperClient(delay=delay, workers=workers) as client:
            result = scrape_all_events(
                session,
                client,
                progress_callback=_scrape_progress,
                after_date=after_date,
                workers=workers,
            )

        table = Table(title="UFCStats Scrape Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green", justify="right")
        table.add_row("Events accepted", str(result.accepted))
        table.add_row("Events updated", str(result.updated))
        table.add_row("Events rejected", str(result.rejected))
        console.print(table)

    except Exception as exc:
        session.rollback()
        console.print(f"[red]Scrape failed: {exc}[/red]")
        raise SystemExit(1) from exc
    finally:
        session.close()


@scrape_app.command("latest")
def scrape_latest(
    delay: Annotated[float, typer.Option(help="Delay between requests in seconds")] = 1.5,
    workers: Annotated[
        int,
        typer.Option(
            help="Concurrent worker threads (default 4). Reduce to 1 if rate-limited (429s).",
        ),
    ] = 4,
) -> None:
    """Scrape only new events from UFCStats.com (incremental update).

    Compares UFCStats event listing against database and only downloads
    events not yet scraped. Fast way to update after a UFC event.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from ufc_prediction.scraper.client import ScraperClient

    console.print("[bold]Checking for new events on UFCStats...[/bold]")
    console.print(f"Workers: {workers}")

    session = SessionLocal()
    try:
        with ScraperClient(delay=delay, workers=workers) as client:
            result = scrape_latest_events(session, client, workers=workers)

        if result.accepted == 0 and result.rejected == 0:
            console.print("[green]Database is up to date. No new events found.[/green]")
        else:
            table = Table(title="Incremental Scrape Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Count", style="green", justify="right")
            table.add_row("New events scraped", str(result.accepted))
            table.add_row("Events rejected", str(result.rejected))
            console.print(table)

    except Exception as exc:
        session.rollback()
        console.print(f"[red]Scrape failed: {exc}[/red]")
        raise SystemExit(1) from exc
    finally:
        session.close()


def _scrape_progress(current: int, total: int) -> None:
    """Progress callback for scrape commands."""
    idx = current + 1
    if idx % 50 == 0 or idx == total:
        console.print(f"  [{idx}/{total}] events processed")


@scrape_app.command("sherdog")
def scrape_sherdog(
    delay: Annotated[float, typer.Option(help="Seconds between requests (min 2.0 per D-04)")] = 2.0,
    dry_run: Annotated[
        bool, typer.Option(help="Search and match only, don't fetch profiles")
    ] = False,
    workers: Annotated[
        int,
        typer.Option(
            help=(
                "Concurrent worker threads (default 2). Each worker "
                "enforces the 2.0s D-04 delay independently."
            ),
        ),
    ] = 2,
    since_year: Annotated[
        int | None,
        typer.Option(
            help=(
                "Only process fighters with a UFC fight on or after "
                "Jan 1 of YYYY. Default: all fighters."
            ),
        ),
    ] = None,
) -> None:
    """Scrape pre-UFC records from Sherdog.com for all fighters in the database.

    Searches Sherdog for each fighter, matches by name using fuzzy matching,
    fetches their full fight history, filters pre-UFC fights, and stores
    computed career stats as JSON on the Fighter model.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from ufc_prediction.scraper.client import ScraperClient

    # Enforce minimum delay per D-04
    effective_delay = max(delay, 2.0)
    if delay < 2.0:
        console.print(f"[yellow]Delay {delay}s below minimum 2.0s, using 2.0s[/yellow]")

    mode = "DRY RUN" if dry_run else "full scrape"
    console.print(f"[bold]Starting Sherdog {mode}...[/bold]")
    console.print(f"Request delay: {effective_delay}s")
    console.print(f"Workers: {workers}")
    if since_year is not None:
        console.print(f"Fighter filter: UFC fight on or after {since_year}-01-01")

    session = SessionLocal()
    try:
        with ScraperClient(delay=effective_delay, workers=workers) as client:
            scraper = SherdogScraper(client, session, workers=workers)
            result = scraper.scrape_all_fighters(
                progress_callback=_sherdog_progress,
                dry_run=dry_run,
                since_year=since_year,
            )

        table = Table(title="Sherdog Scrape Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green", justify="right")
        table.add_row("Matched", str(result["matched"]))
        table.add_row("Unmatched", str(result["unmatched"]))
        table.add_row("Skipped (already done)", str(result["skipped"]))
        table.add_row("Errors", str(result["errors"]))
        console.print(table)

        # Print unmatched fighters for manual review (D-08)
        unmatched_names = result.get("unmatched_names", [])
        if unmatched_names:
            console.print(f"\n[yellow]Unmatched fighters ({len(unmatched_names)}):[/yellow]")
            for name in unmatched_names:
                console.print(f"  - {name}")

    except Exception as exc:
        session.rollback()
        console.print(f"[red]Sherdog scrape failed: {exc}[/red]")
        raise SystemExit(1) from exc
    finally:
        session.close()


def _sherdog_progress(current: int, total: int) -> None:
    """Progress callback for Sherdog scrape command."""
    idx = current + 1
    if idx % 25 == 0 or idx == total:
        console.print(f"  [{idx}/{total}] fighters processed")


# ── BFO (BestFightOdds) Scrape + Ingest ─────────────────────────────────────

# T-13-02-03 (path traversal mitigation): absolute paths under these prefixes
# are considered untrusted filesystem locations; the command refuses to use
# them as data folders.
_BFO_FORBIDDEN_PATH_PREFIXES = (
    Path("/etc"),
    Path("/sys"),
    Path("/proc"),
    Path("/root"),
)


def _validate_bfo_data_folder(data_folder: Path) -> Path:
    """Resolve + sanity-check a user-supplied ``--data-folder``.

    Raises ``SystemExit(1)`` on paths under the forbidden prefixes.
    """
    resolved = data_folder.expanduser().resolve()
    if any(str(resolved).startswith(str(fp)) for fp in _BFO_FORBIDDEN_PATH_PREFIXES):
        console.print(f"[red]Refusing to use unsafe data folder: {resolved}[/red]")
        raise SystemExit(1)
    return resolved


@scrape_app.command("odds")
def scrape_odds(
    delay: Annotated[float, typer.Option(help="Seconds between BFO requests per session")] = 2.0,
    n_sessions: Annotated[
        int,
        typer.Option(
            help=(
                "DEPRECATED — kept for CLI backward-compatibility. "
                "BFOScraper uses workers=4 (D-03)."
            )
        ),
    ] = 2,
    min_date: Annotated[str, typer.Option(help="Earliest event date YYYY-MM-DD")] = "2007-08-01",
    data_folder: Annotated[
        Path,
        typer.Option(
            help=("Where the BFOScraper writes CSVs (and where the ingester reads)."),
        ),
    ] = Path("data/bfo"),
    skip_scrape: Annotated[
        bool,
        typer.Option(
            help=("Skip the BFOScraper network scrape; only ingest existing CSVs."),
        ),
    ] = False,
) -> None:
    """Scrape BestFightOdds and ingest into the fight_odds table.

    Flow (Phase 15):
      1. Unless --skip-scrape, invoke our in-house BFOScraper (D-01)
         which writes CSVs to ``data_folder``. This replaces the
         Python 3.14-broken ufcscraper odds-scraper component.
      2. Run BFOOddsIngester to validate CSVs, fuzzy-match fighter
         names, apply proportional vig-removal (D-02), and upsert to
         the ``fight_odds`` table (idempotent).
      3. Print a Rich summary with coverage percentage; emit a warning
         if coverage falls below 60 percent (Pitfall 7 / A2 assumption).
    """
    import datetime as _dt

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # T-13-02-03: path-traversal mitigation.
    resolved = _validate_bfo_data_folder(data_folder)
    resolved.mkdir(parents=True, exist_ok=True)

    # n_sessions retained for CLI backward-compat; new BFOScraper uses
    # workers=4 (D-03) — the parameter no longer wires through but we
    # keep the flag visible so existing scripts/cron lines don't break.
    _ = n_sessions

    if not skip_scrape:
        try:
            parsed_min_date = _dt.datetime.strptime(min_date, "%Y-%m-%d").date()
        except ValueError as exc:
            console.print(f"[red]Invalid --min-date {min_date!r}: {exc}[/red]")
            raise SystemExit(1) from exc

        console.print(f"[bold]Scraping BestFightOdds to {resolved}...[/bold]")
        console.print(f"  Delay: {delay}s · Workers: 4 · Min date: {min_date}")
        client = ScraperClient(delay=delay, workers=4)
        scrape_session = SessionLocal()
        try:
            scraper = BFOScraper(
                client=client,
                session=scrape_session,
                data_folder=resolved,
                workers=4,
            )
            scrape_summary = scraper.scrape_all(min_date=parsed_min_date)
        finally:
            client.close()
            scrape_session.close()
        console.print(
            f"  Scraped {scrape_summary.fighters_matched} matched / "
            f"{scrape_summary.fighters_processed} fighters, "
            f"{scrape_summary.fights_emitted} fight rows. "
            f"Parse errors: {scrape_summary.parse_errors}, "
            f"CAPTCHA hits: {scrape_summary.captcha_hits}."
        )
        if scrape_summary.captcha_hits > 0:
            console.print(
                "[yellow]CAPTCHA detected on at least one BFO page — "
                "consider increasing --delay or reducing concurrency "
                "before next run.[/yellow]"
            )
    else:
        console.print(f"[yellow]--skip-scrape[/yellow] — reading existing CSVs from {resolved}")

    from ufc_prediction.scraper.bfo_ingest import BFOOddsIngester

    session = SessionLocal()
    try:
        ingester = BFOOddsIngester(session, data_folder=resolved)
        summary = ingester.ingest_all()
    except Exception as exc:
        session.rollback()
        console.print(f"[red]BFO ingest failed: {exc}[/red]")
        raise SystemExit(1) from exc
    finally:
        session.close()

    table = Table(title="BFO Ingest Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green", justify="right")
    table.add_row("BFO fights scanned", str(summary.bfo_fights_scanned))
    table.add_row("Fights matched", str(summary.fights_matched))
    table.add_row("Fighters matched", str(summary.fighters_matched))
    table.add_row("Fighters unmatched", str(summary.fighters_unmatched))
    table.add_row("Rows upserted", str(summary.rows_upserted))
    table.add_row("Coverage %", f"{summary.coverage_pct * 100:.1f}%")
    console.print(table)

    if summary.coverage_pct < 0.60:
        console.print(
            f"[yellow]Warning: coverage below 60% "
            f"({summary.coverage_pct * 100:.1f}%). "
            "Model feature signal may be weak (see Pitfall 7).[/yellow]"
        )

    if summary.unmatched_bfo_names:
        console.print(
            f"\n[yellow]Unmatched BFO fighters ({len(summary.unmatched_bfo_names)}):[/yellow]"
        )
        for name in summary.unmatched_bfo_names[:20]:
            console.print(f"  - {name}")
        if len(summary.unmatched_bfo_names) > 20:
            console.print(f"  ... and {len(summary.unmatched_bfo_names) - 20} more")


# ── Phase 28 INGEST-V23 Commands (referee + venue backfill) ─────────────────


# scripts/ is sibling to src/; add it to sys.path so we can import the slim
# driver without packaging it. The driver lives in scripts/ (not src/) per
# Phase 22 convention — one-off + slim driver scripts colocate with other
# audit drivers there.
def _ensure_scripts_on_path() -> None:
    import sys as _sys

    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    if str(scripts_dir) not in _sys.path:
        _sys.path.insert(0, str(scripts_dir))


_ensure_scripts_on_path()
# Module-level import so unittest.mock.patch("...scripts.scrape_referees_full.main")
# can resolve and substitute the driver entry-point during tests.
# Use importlib because `scripts/` lacks __init__.py and isn't a real package.
import importlib as _importlib  # noqa: E402

_scrape_referees_full = _importlib.import_module("scrape_referees_full")


@scrape_app.command("referees")
def scrape_referees_cmd(
    delay: Annotated[
        float,
        typer.Option(help="Seconds between UFCStats requests (Phase 22 convention: 1.2s)"),
    ] = 1.2,
    workers: Annotated[
        int,
        typer.Option(help="Concurrent worker threads (default 4)"),
    ] = 4,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Print events_to_process + cache_hits_expected + ETA without network/DB writes",
        ),
    ] = False,
    confirm: Annotated[
        bool,
        typer.Option(
            "--confirm",
            help="Required to actually run (CONTEXT D-08 safety gate; no default-confirm)",
        ),
    ] = False,
) -> None:
    """Backfill events.referee_id idempotently across the corpus (INGEST-V23-01).

    Per CONTEXT D-08: pass --dry-run to preview ETA + counts without
    network calls. Pass --confirm to actually scrape. Re-runs are idempotent
    (CR-01 binding from Phase 22: never overwrites a previously-set
    referee_id).
    """
    exit_code = _scrape_referees_full.main(
        delay=delay,
        workers=workers,
        dry_run=dry_run,
        confirm=confirm,
    )
    if exit_code != 0:
        raise typer.Exit(exit_code)


@scrape_app.command("venues")
def scrape_venues_cmd(
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print counts only; no DB writes"),
    ] = False,
    confirm: Annotated[
        bool,
        typer.Option(
            "--confirm",
            help="Required to actually write (CONTEXT D-08 safety gate)",
        ),
    ] = False,
    unmatched_output: Annotated[
        Path,
        typer.Option(help="28-UNMATCHED-VENUES.md output path"),
    ] = Path(".planning/phases/28-referee-venue-ingestion-pipeline/28-UNMATCHED-VENUES.md"),
    venues_csv: Annotated[
        Path,
        typer.Option(help="venues.csv to load (post-Plan-28-01 expanded form)"),
    ] = Path("data/venues.csv"),
) -> None:
    """Backfill events.venue_id via venues.csv lookup + fuzzy match (INGEST-V23-02).

    Three-tier per CONTEXT D-06: exact -> rapidfuzz.token_sort_ratio>=85 ->
    UNRESOLVED. Unmatched locations are surfaced to 28-UNMATCHED-VENUES.md
    for operator review (operator manually adds rows to data/venues.csv with
    lat/lon/timezone_iana, then re-runs `ufc scrape venues --confirm`).
    Re-runs are idempotent (assign_venue_id_if_null guards on
    events.venue_id IS NULL — CR-01 carry-forward from referees).
    """
    import csv as _csv
    import os as _os

    from sqlalchemy import select

    from ufc_prediction.models.event import Event
    from ufc_prediction.scraper.venue_match import (
        _normalize_location,
        assign_venue_id_if_null,
        match_venue,
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load venues.csv -> (lookup, names_raw)
    if not venues_csv.exists():
        console.print(f"[red]venues.csv not found at {venues_csv}[/red]")
        raise typer.Exit(2)
    venue_lookup: dict[str, int] = {}
    venue_names_raw: list[tuple[int, str]] = []
    with venues_csv.open("r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            vid = int(row["venue_id"])
            raw = row["name"]
            venue_lookup[_normalize_location(raw)] = vid
            venue_names_raw.append((vid, raw))
    n_venues = len(venue_names_raw)

    session = SessionLocal()
    try:
        # SELECT events WHERE venue_id IS NULL AND location IS NOT NULL.
        events = (
            session.execute(
                select(Event)
                .where(Event.venue_id.is_(None))
                .where(Event.location.isnot(None))
                .order_by(Event.id)
            )
            .scalars()
            .all()
        )
        events_to_process = len(events)

        # Pre-flight: count unresolvable distinct strings (dry-run pass).
        distinct_locations: dict[str, int] = {}
        for ev in events:
            distinct_locations[ev.location] = distinct_locations.get(ev.location, 0) + 1
        unresolvable_distinct = 0
        for loc in distinct_locations:
            if match_venue(loc, venue_lookup, venue_names_raw) is None:
                unresolvable_distinct += 1

        if events_to_process == 0:
            console.print(
                "[green]Substrate already complete — 0 events with NULL venue_id. "
                "Exiting 0.[/green]"
            )
            return

        if dry_run:
            console.print("[bold]DRY-RUN preview (no DB writes)[/bold]")
            console.print(f"Events to process: {events_to_process}")
            console.print(f"Distinct locations: {len(distinct_locations)}")
            console.print(f"Unresolvable distinct strings: {unresolvable_distinct}")
            console.print(f"Venues loaded from {venues_csv}: {n_venues}")
            console.print("Re-invoke with --confirm to actually write FKs.")
            return

        if not confirm:
            console.print(
                "[yellow]SAFETY GATE — re-invoke with --confirm to actually "
                "write venue FKs, or --dry-run for a preview (CONTEXT D-08). "
                "Refusing to proceed without explicit confirmation.[/yellow]"
            )
            raise typer.Exit(1)

        # Confirmed path — assign FKs idempotently.
        resolved_exact = 0
        resolved_fuzzy = 0
        preserved = 0
        unmatched_counter: dict[str, int] = {}
        for ev in events:
            vid, kind = assign_venue_id_if_null(
                session,
                event_id=ev.id,
                location=ev.location,
                venue_lookup=venue_lookup,
                venue_names_raw=venue_names_raw,
            )
            if kind == "exact":
                resolved_exact += 1
            elif kind is not None and kind.startswith("fuzzy:"):
                resolved_fuzzy += 1
            elif kind == "preserved":
                preserved += 1
            else:
                unmatched_counter[ev.location] = unmatched_counter.get(ev.location, 0) + 1
            session.commit()  # per-event commit (Phase 22 convention)

        # Emit 28-UNMATCHED-VENUES.md (atomic; WR-05).
        if unmatched_counter:
            _emit_unmatched_venues_md(unmatched_output, unmatched_counter)
            console.print(
                f"[yellow]{len(unmatched_counter)} unmatched location(s) — "
                f"see {unmatched_output}[/yellow]"
            )

        console.print(
            f"[green]venue backfill complete.[/green] "
            f"exact={resolved_exact}, fuzzy={resolved_fuzzy}, "
            f"preserved={preserved}, unmatched={len(unmatched_counter)}"
        )
    finally:
        session.close()
    # Silence unused import warning for os (used inside _emit_unmatched_venues_md
    # via os.replace for WR-05 atomic write).
    _ = _os


def _emit_unmatched_venues_md(output_path: Path, unmatched: dict[str, int]) -> None:
    """Atomic WR-05 write of 28-UNMATCHED-VENUES.md.

    Sort by descending n_events so the operator works the biggest gaps first.
    """
    import os as _os
    from datetime import datetime as _dt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Phase 28 — Unmatched Venues (operator review)",
        "",
        f"**Generated:** {_dt.now(UTC).isoformat()}",
        "**Match tiers attempted:** exact, fuzz.token_sort_ratio>=85",
        "**Action:** operator manually adds rows to data/venues.csv with "
        "lat/lon/timezone_iana, then re-runs `ufc scrape venues --confirm`.",
        "",
        "| n_events | location |",
        "|---:|---|",
    ]
    for loc, n in sorted(unmatched.items(), key=lambda kv: (-kv[1], kv[0])):
        # Escape pipes inside location to keep markdown table well-formed.
        safe_loc = loc.replace("|", r"\|")
        lines.append(f"| {n} | {safe_loc} |")
    lines.append("")  # trailing newline
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text("\n".join(lines), encoding="utf-8")
    _os.replace(tmp_path, output_path)


# ── Elo Commands ─────────────────────────────────────────────────────────────


@elo_app.command("compute")
def compute_elo(
    domain: Annotated[
        bool, typer.Option(help="Compute domain-specific (striking/grappling) Elo")
    ] = True,
) -> None:
    """Compute Elo ratings for all fighters and write snapshots to DB.

    Loads all fights chronologically, runs the EloEngine, and flushes
    snapshots to the elo_snapshots table (idempotent).

    With --domain (default), also runs a second pass computing
    striking and grappling Elo from round-by-round stats (D-09).
    """
    console.print("[bold]Computing Elo ratings...[/bold]")
    session = SessionLocal()
    try:
        fights = load_fights_chronological(session)
        console.print(f"Loaded {len(fights)} fights")

        # DEBUT-V25-03 (Phase 43): seeded debutant init. load_seeds returns
        # {} if the CSV is missing -> EloEngine then degrades to pre-Phase-43
        # flat-1500 default behavior (Test 3 bit-exact regression invariant).
        seeds = load_seeds(_SHERDOG_PRE_UFC_CSV)
        n_seeded = len(seeds)
        dispatch_note = (
            "will dispatch seeds on first-encounter"
            if n_seeded
            else "empty -- falling back to flat 1500 default"
        )
        console.print(
            f"Loaded {n_seeded} debutant Elo seeds from {_SHERDOG_PRE_UFC_CSV} ({dispatch_note})"
        )
        engine = EloEngine(EloConfig(), seeds=seeds)
        start = time.perf_counter()
        snapshots = engine.compute_all(fights)
        elapsed = time.perf_counter() - start

        count = flush_snapshots(session, snapshots)
        session.commit()
        unique_fighters = len({s.fighter_id for s in snapshots})
        console.print(f"Wrote {count} snapshots in {elapsed:.2f}s")
        console.print(f"Unique fighters rated: {unique_fighters}")
        # DEBUT-V25-03 dispatch counter — operator visibility into the
        # seed-vs-default population (matches T-43-03-05 mitigation).
        seeded_fighters_in_corpus = sum(
            1 for fid in {s.fighter_id for s in snapshots} if fid in seeds
        )
        fallback_fighters = unique_fighters - seeded_fighters_in_corpus
        console.print(
            f"Seeded debutants in corpus: {seeded_fighters_in_corpus} | "
            f"Fallback-to-1500: {fallback_fighters}"
        )

        if domain:
            console.print("[bold]Computing domain Elo (striking/grappling)...[/bold]")
            round_stats = load_round_stats_by_fight(session)
            domain_computer = DomainEloComputer(EloConfig())
            domain_start = time.perf_counter()
            domain_snaps = domain_computer.compute_all(fights, snapshots, round_stats)
            domain_elapsed = time.perf_counter() - domain_start

            domain_count = flush_domain_snapshots(session, domain_snaps)
            session.commit()
            fights_with_domain = len({s.fight_id for s in domain_snaps}) // 2
            console.print(
                f"Wrote {domain_count} domain snapshots in {domain_elapsed:.2f}s "
                f"({fights_with_domain} fights with round data)"
            )
    except Exception as exc:
        session.rollback()
        console.print(f"[red]Error computing Elo ratings: {exc}[/red]")
        raise SystemExit(1) from exc
    finally:
        session.close()


@elo_app.command("backtest")
def backtest_elo(
    cutoff_date: Annotated[str, typer.Option(help="YYYY-MM-DD temporal split date")],
) -> None:
    """Run K-factor grid search and evaluate configs by Brier score.

    Splits fights at the cutoff date: training before, test on/after.
    Evaluates 225 K-factor configurations and displays the top 10.
    """
    parsed_date = date_type.fromisoformat(cutoff_date)

    session = SessionLocal()
    try:
        fights = load_fights_chronological(session)
    finally:
        session.close()

    configs = generate_grid_configs()
    test_count = sum(1 for f in fights if f.event_date >= parsed_date)
    console.print(f"Running backtest with {len(configs)} configurations...")
    console.print(f"Cutoff date: {cutoff_date}, Test fights: {test_count}")

    results = run_backtest(fights, configs, parsed_date)

    # Display top 10 results
    table = Table(title="Top 10 Backtest Results")
    table.add_column("Rank", style="cyan", justify="right")
    table.add_column("K_initial", justify="right")
    table.add_column("K_experienced", justify="right")
    table.add_column("K_transition", justify="right")
    table.add_column("Brier Score", style="green", justify="right")
    table.add_column("N Test Fights", justify="right")

    for i, r in enumerate(results[:10], 1):
        cfg = r["config"]
        table.add_row(
            str(i),
            f"{cfg.k_initial:.0f}",
            f"{cfg.k_experienced:.0f}",
            str(cfg.k_transition_fights),
            f"{r['brier_score']:.6f}",
            str(r["n_test_fights"]),
        )

    console.print(table)

    if results:
        best = results[0]
        best_cfg = best["config"]
        console.print(
            f"\n[bold green]Best config:[/bold green] "
            f"K_init={best_cfg.k_initial:.0f}, "
            f"K_exp={best_cfg.k_experienced:.0f}, "
            f"K_trans={best_cfg.k_transition_fights}, "
            f"Brier={best['brier_score']:.6f}"
        )


def _print_backtest_results(
    title: str,
    results: list[dict],
    top: int,
    cols: list[str],
    extractors: list,
) -> None:
    """Print a Rich table of backtest results."""
    table = Table(title=f"{title} Results (Top {min(top, len(results))})")
    table.add_column("Rank", style="cyan", justify="right")
    for col in cols:
        table.add_column(col, justify="right")
    table.add_column("Brier Score", style="green", justify="right")
    table.add_column("N Test", justify="right")

    for i, r in enumerate(results[:top], 1):
        cfg = r["config"]
        row = [str(i)]
        for extractor in extractors:
            row.append(extractor(cfg))
        row.append(f"{r['brier_score']:.6f}")
        row.append(str(r["n_test_fights"]))
        table.add_row(*row)

    console.print(table)

    if results:
        best = results[0]
        console.print(f"  [bold green]Best Brier: {best['brier_score']:.6f}[/bold green]")


@elo_app.command("backtest-params")
def backtest_params(
    cutoff_date: Annotated[str, typer.Option(help="YYYY-MM-DD temporal split date")],
    group: Annotated[
        str,
        typer.Option(help="Parameter group: mov, transfer, inactivity, ewma, domain, joint, all"),
    ] = "all",
    top: Annotated[int, typer.Option(help="Number of top results to display")] = 10,
    output_dir: Annotated[str, typer.Option(help="Directory for JSON results")] = "results",
) -> None:
    """Run parameter validation backtests and save results as JSON (per D-11).

    Supports individual parameter groups or 'all' to run the full staged
    approach (D-01): independent backtests first, then joint optimization.
    """
    parsed_date = date_type.fromisoformat(cutoff_date)
    results_path = Path(output_dir)

    session = SessionLocal()
    try:
        fights = load_fights_chronological(session)
        console.print(f"Loaded {len(fights)} fights")

        groups_to_run = (
            ["mov", "transfer", "inactivity", "ewma", "domain", "joint"]
            if group == "all"
            else [group]
        )

        independent_results: dict[str, list[dict]] = {}

        for grp in groups_to_run:
            if grp == "mov":
                configs = generate_mov_grid()
                console.print(f"\n[bold]MOV Multiplier Backtest ({len(configs)} configs)...[/bold]")
                results = run_backtest(fights, configs, parsed_date)
                independent_results["mov"] = results
                _print_backtest_results(
                    "MOV Multipliers",
                    results,
                    top,
                    cols=["KO/TKO", "Submission", "Split"],
                    extractors=[
                        lambda c: f"{c.mov_ko_tko:.1f}",
                        lambda c: f"{c.mov_submission:.1f}",
                        lambda c: f"{c.mov_split:.1f}",
                    ],
                )
                save_backtest_results(
                    "mov_multipliers",
                    results,
                    parsed_date,
                    {"ko_tko": "1.2-1.8", "submission": "1.1-1.7", "split": "0.5-1.0"},
                    output_dir=results_path,
                )

            elif grp == "transfer":
                configs = generate_transfer_grid()
                console.print(
                    f"\n[bold]Division Transfer Backtest ({len(configs)} configs)...[/bold]"
                )
                results = run_backtest(fights, configs, parsed_date)
                independent_results["transfer"] = results
                _print_backtest_results(
                    "Division Transfer",
                    results,
                    top,
                    cols=["Transfer %"],
                    extractors=[lambda c: f"{c.division_transfer_pct:.0%}"],
                )
                save_backtest_results(
                    "transfer",
                    results,
                    parsed_date,
                    {"transfer_pct": "0.50-1.00"},
                    output_dir=results_path,
                )

            elif grp == "inactivity":
                configs = generate_inactivity_grid()
                console.print(
                    f"\n[bold]Inactivity Regression Backtest ({len(configs)} configs)...[/bold]"
                )
                results = run_backtest(fights, configs, parsed_date)
                independent_results["inactivity"] = results
                _print_backtest_results(
                    "Inactivity Regression",
                    results,
                    top,
                    cols=["Threshold (days)", "Rate", "Cap"],
                    extractors=[
                        lambda c: str(c.inactivity_threshold_days),
                        lambda c: f"{c.inactivity_regression_rate:.2f}",
                        lambda c: f"{c.inactivity_regression_cap:.2f}",
                    ],
                )
                save_backtest_results(
                    "inactivity",
                    results,
                    parsed_date,
                    {"threshold_days": "270-548", "rate": "0.05-0.20", "cap": "0.30-0.75"},
                    output_dir=results_path,
                )

            elif grp == "ewma":
                round_stats = load_round_stats_by_fight(session)
                console.print("\n[bold]EWMA Half-Life Backtest (4 configs)...[/bold]")

                from ufc_prediction.features.queries import load_all_domain_elo

                domain_elo = load_all_domain_elo(session)

                ewma_configs = generate_ewma_grid()
                ewma_results = []
                for ec in ewma_configs:
                    console.print(f"  Evaluating half_life={ec['half_life']}...")
                    result = evaluate_ewma_brier(
                        ec["half_life"],
                        ec["alpha"],
                        fights,
                        round_stats,
                        domain_elo,
                        parsed_date,
                    )
                    ewma_results.append(result)
                ewma_results.sort(key=lambda r: r["brier_score"])

                table = Table(title=f"EWMA Half-Life Results (Top {min(top, len(ewma_results))})")
                table.add_column("Rank", style="cyan", justify="right")
                table.add_column("Half-Life", justify="right")
                table.add_column("Alpha", justify="right")
                table.add_column("Brier Score", style="green", justify="right")
                table.add_column("N Test", justify="right")
                for i, r in enumerate(ewma_results[:top], 1):
                    table.add_row(
                        str(i),
                        str(r["half_life"]),
                        f"{r['alpha']:.4f}",
                        f"{r['brier_score']:.6f}",
                        str(r["n_test_fights"]),
                    )
                console.print(table)

                save_backtest_results(
                    "ewma",
                    ewma_results,
                    parsed_date,
                    {"half_life": "2-5"},
                    output_dir=results_path,
                )

            elif grp == "domain":
                round_stats = load_round_stats_by_fight(session)
                console.print("\n[bold]Domain Attribution Backtest (25 configs)...[/bold]")

                # noqa: DEBUT-V25 -- backtest path intentionally unseeded for
                # v2.3-baseline comparability. Phase 43 backtest-vs-prod
                # comparisons need a constant 1500-default substrate so the
                # backtest grid search isolates domain-attribution params from
                # debutant-seed effects. compute_elo (the canonical path at
                # ~line 961) IS seeded.
                engine = EloEngine(EloConfig())
                overall_snaps = engine.compute_all(fights)

                domain_configs = generate_domain_attribution_grid()
                domain_results = []
                for dc in domain_configs:
                    result = evaluate_domain_brier(
                        dc,
                        fights,
                        overall_snaps,
                        round_stats,
                        parsed_date,
                    )
                    domain_results.append(result)
                domain_results.sort(key=lambda r: r["brier_score"])

                table = Table(
                    title=f"Domain Attribution Results (Top {min(top, len(domain_results))})"
                )
                table.add_column("Rank", style="cyan", justify="right")
                table.add_column("KO Str/Grap", justify="right")
                table.add_column("Sub Str/Grap", justify="right")
                table.add_column("Brier Score", style="green", justify="right")
                table.add_column("N Test", justify="right")
                for i, r in enumerate(domain_results[:top], 1):
                    c = r["config"]
                    table.add_row(
                        str(i),
                        f"{c['ko_striking']:.1f}/{c['ko_grappling']:.1f}",
                        f"{c['sub_striking']:.1f}/{c['sub_grappling']:.1f}",
                        f"{r['brier_score']:.6f}",
                        str(r["n_test_fights"]),
                    )
                console.print(table)

                save_backtest_results(
                    "domain_attribution",
                    domain_results,
                    parsed_date,
                    {"ko_striking": "0.6-1.0", "sub_grappling": "0.6-1.0"},
                    output_dir=results_path,
                )

            elif grp == "joint":
                if not independent_results:
                    # Try to load from disk
                    for key in ["mov", "transfer", "inactivity"]:
                        fname = "mov_multipliers" if key == "mov" else key
                        fpath = results_path / f"backtest_{fname}.json"
                        if fpath.exists():
                            loaded = load_backtest_results(fpath)
                            reconstructed = []
                            for r in loaded["results"]:
                                cfg_dict = r.get("config", {})
                                valid_fields = set(EloConfig.__dataclass_fields__)
                                filtered = {k: v for k, v in cfg_dict.items() if k in valid_fields}
                                reconstructed.append(
                                    {
                                        "config": EloConfig(**filtered),
                                        "brier_score": r["brier_score"],
                                        "n_test_fights": r.get("n_test_fights", 0),
                                    }
                                )
                            independent_results[key] = reconstructed

                console.print("\n[bold]Joint Optimization (top-3 cross-product)...[/bold]")
                joint_results = run_joint_optimization(
                    fights,
                    independent_results,
                    parsed_date,
                    top_n=3,
                )
                if joint_results:
                    _print_backtest_results(
                        "Joint Optimization",
                        joint_results,
                        top,
                        cols=[
                            "KO/TKO",
                            "Sub",
                            "Split",
                            "Transfer",
                            "Inact.Thresh",
                            "Inact.Rate",
                            "Inact.Cap",
                        ],
                        extractors=[
                            lambda c: f"{c.mov_ko_tko:.1f}",
                            lambda c: f"{c.mov_submission:.1f}",
                            lambda c: f"{c.mov_split:.1f}",
                            lambda c: f"{c.division_transfer_pct:.0%}",
                            lambda c: str(c.inactivity_threshold_days),
                            lambda c: f"{c.inactivity_regression_rate:.2f}",
                            lambda c: f"{c.inactivity_regression_cap:.2f}",
                        ],
                    )
                    save_backtest_results(
                        "joint",
                        joint_results,
                        parsed_date,
                        {"strategy": "top-3 cross-product from independent sweeps"},
                        output_dir=results_path,
                    )
                else:
                    console.print(
                        "[yellow]No independent results available for joint optimization.[/yellow]"
                    )

    except Exception as exc:
        console.print(f"[red]Error during backtest: {exc}[/red]")
        raise SystemExit(1) from exc
    finally:
        session.close()

    console.print(f"\n[bold green]Results saved to {results_path}/[/bold green]")


# ── Fighter Commands ────────────────────────────────────────────────────────


@fighter_app.command("lookup")
def fighter_lookup(
    name: Annotated[str, typer.Argument(help="Fighter name to search for")],
) -> None:
    """Look up a fighter's Elo rating, division, and record."""
    session = SessionLocal()
    try:
        matches = search_fighters(session, name)

        if not matches:
            console.print(f"[yellow]No fighters found matching '{name}'[/yellow]")
            return

        # DEDUP-03 (Phase 14): collapse same-real-world-person duplicates
        # before deciding whether to show the multi-match table. Group by
        # exact lowercase name (matching _resolve_fighter's matching rule)
        # then pick the canonical row per group via source priority.
        by_name: dict[str, list[Fighter]] = {}
        for f in matches:
            by_name.setdefault(f.name.lower(), []).append(f)
        deduped = [_prefer_canonical(group) for group in by_name.values()]

        if len(deduped) > 1:
            # D-02: genuinely different real people -- show numbered list
            table = Table(title=f"Multiple fighters matching '{name}'")
            table.add_column("#", style="cyan", justify="right")
            table.add_column("Name", style="bold")
            table.add_column("Division")
            table.add_column("Elo", justify="right")
            for i, fighter in enumerate(deduped, 1):
                detail = get_fighter_detail(session, fighter.id, division=None)
                elo_str = f"{detail['elo']:.0f}" if detail["elo"] else "N/A"
                div_str = detail["division"] or "Unknown"
                table.add_row(str(i), fighter.name, div_str, elo_str)
            console.print(table)
            return

        # D-04: exactly one real-world person -- show full details
        fighter = deduped[0]
        divisions = get_fighter_divisions(session, fighter.id)

        if not divisions:
            console.print(f"[bold]{fighter.name}[/bold]")
            console.print("[yellow]No rated fights found.[/yellow]")
            return

        console.print(f"\n[bold]{fighter.name}[/bold]\n")
        for div in divisions:
            detail = get_fighter_detail(session, fighter.id, div)
            table = Table(title=f"{div}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="bold")
            table.add_row(
                "Elo Rating",
                f"{detail['elo']:.0f}" if detail["elo"] else "N/A",
            )
            # Domain Elo (D-11)
            domain_elo = get_fighter_domain_elo(session, fighter.id, div)
            if domain_elo["striking_elo"] is not None:
                table.add_row(
                    "Striking Elo",
                    f"{domain_elo['striking_elo']:.0f}",
                )
            if domain_elo["grappling_elo"] is not None:
                table.add_row(
                    "Grappling Elo",
                    f"{domain_elo['grappling_elo']:.0f}",
                )
            table.add_row("Division", detail["division"])
            table.add_row("Record", f"{detail['wins']}W - {detail['losses']}L")
            table.add_row("Total Fights", str(detail["total_fights"]))
            table.add_row(
                "Last Fight",
                str(detail["last_fight_date"]) if detail["last_fight_date"] else "N/A",
            )
            console.print(table)
            console.print()

    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise SystemExit(1) from exc
    finally:
        session.close()


@fighter_app.command("rankings")
def fighter_rankings(
    weight_class: Annotated[str, typer.Argument(help="Weight class (e.g., 'Lightweight')")],
    top: Annotated[int, typer.Option(help="Number of fighters to display")] = 15,
) -> None:
    """Display top fighters in a division ranked by Elo."""
    if top < 1:
        console.print("[red]--top must be at least 1[/red]")
        raise SystemExit(1)

    # D-09: resolve partial weight class match
    matches = resolve_weight_class(weight_class)

    if not matches:
        console.print(f"[yellow]No weight class matching '{weight_class}'[/yellow]")
        console.print("Valid divisions: " + ", ".join(sorted(list_divisions())))
        return

    if len(matches) > 1:
        console.print(f"[yellow]Ambiguous weight class '{weight_class}'. Did you mean:[/yellow]")
        for m in matches:
            console.print(f"  - {m}")
        return

    division = matches[0]
    session = SessionLocal()
    try:
        rankings = get_division_rankings(session, division, limit=top)

        if not rankings:
            console.print(f"[yellow]No ranked fighters in {division}[/yellow]")
            return

        table = Table(title=f"{division} Rankings (Top {top})")
        table.add_column("Rank", style="cyan", justify="right")
        table.add_column("Fighter", style="bold")
        table.add_column("Elo", style="green", justify="right")
        table.add_column("Fights", justify="right")
        table.add_column("Last Active", justify="right")

        for i, row in enumerate(rankings, 1):
            table.add_row(
                str(i),
                row["name"],
                f"{row['elo']:.0f}",
                str(row["fights"]),
                str(row["last_date"]),
            )

        console.print(table)

    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise SystemExit(1) from exc
    finally:
        session.close()


# ── Feature Commands ────────────────────────────────────────────────────────


@features_app.command("compute")
def compute_features() -> None:
    """Compute derived features for all fighters and store in DB.

    Processes all fights chronologically, computing rate features,
    EWMA trends, opponent-adjusted rates, style tags, and PCA embeddings.
    Idempotent -- recomputing overwrites existing feature rows (D-15).
    """
    from ufc_prediction.features.compute import FeatureComputer
    from ufc_prediction.features.config import FeatureConfig
    from ufc_prediction.features.queries import (
        bulk_insert_features,
        flush_features,
        load_all_domain_elo,
        load_all_round_stats,
        load_fights_with_duration,
    )

    console.print("[bold]Computing fighter features...[/bold]")
    session = SessionLocal()
    try:
        config = FeatureConfig()
        fights = load_fights_with_duration(session)
        console.print(f"Loaded {len(fights)} fights")

        round_stats = load_all_round_stats(session)
        console.print(f"Loaded round stats for {len(round_stats)} fights")

        domain_elo = load_all_domain_elo(session)
        console.print(f"Loaded {len(domain_elo)} domain Elo entries")

        computer = FeatureComputer(config)
        start = time.perf_counter()
        feature_rows = computer.compute_all(fights, round_stats, domain_elo)
        elapsed = time.perf_counter() - start

        # Flush and insert (idempotent per D-15)
        flush_features(session, config.feature_set_version)
        count = bulk_insert_features(session, feature_rows)
        session.commit()

        unique_fighters = len({r["fighter_id"] for r in feature_rows})
        console.print(
            f"Computed {count} feature vectors for {unique_fighters} fighters in {elapsed:.2f}s"
        )
    except Exception as exc:
        session.rollback()
        console.print(f"[red]Error computing features: {exc}[/red]")
        raise SystemExit(1) from exc
    finally:
        session.close()


# ── Matchup Commands ───────────────────────────────────────────────────────


def _format_diff(val: float | None) -> str:
    """Format a differential value for display.

    Returns "N/A" for None, "+X.X" for positive (A advantage),
    "X.X" for negative (B advantage) per D-03.
    """
    if val is None:
        return "N/A"
    if val >= 0:
        return f"+{val:.1f}"
    return f"{val:.1f}"


def _format_val(val: float | None, fmt: str = ".1f") -> str:
    """Format a numeric value or return N/A for None."""
    if val is None:
        return "N/A"
    return f"{val:{fmt}}"


def _format_pct(val: float | None) -> str:
    """Format a 0-1 float as percentage or return N/A."""
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


@matchup_app.callback(invoke_without_command=True)
def matchup_compare(
    fighter_a: Annotated[str, typer.Argument(help="Fighter A name or substring")],
    fighter_b: Annotated[str, typer.Argument(help="Fighter B name or substring")],
) -> None:
    """Compare two fighters across physical, Elo, striking, grappling, and style dimensions."""
    from ufc_prediction.matchup.compare import build_matchup

    session = SessionLocal()
    try:
        # Resolve fighter A — apply same DEDUP-03 dedup as fighter_lookup
        matches_a = search_fighters(session, fighter_a)
        if not matches_a:
            console.print(f"[red]No fighters found matching '{fighter_a}'[/red]")
            return
        by_name_a: dict[str, list[Fighter]] = {}
        for f in matches_a:
            by_name_a.setdefault(f.name.lower(), []).append(f)
        deduped_a = [_prefer_canonical(g) for g in by_name_a.values()]
        if len(deduped_a) > 1:
            table = Table(title=f"Multiple matches for '{fighter_a}'")
            table.add_column("#", style="cyan", justify="right")
            table.add_column("Name", style="bold")
            for i, f in enumerate(deduped_a, 1):
                table.add_row(str(i), f.name)
            console.print(table)
            console.print("[yellow]Multiple matches -- please be more specific[/yellow]")
            return

        # Resolve fighter B — apply same DEDUP-03 dedup as fighter_lookup
        matches_b = search_fighters(session, fighter_b)
        if not matches_b:
            console.print(f"[red]No fighters found matching '{fighter_b}'[/red]")
            return
        by_name_b: dict[str, list[Fighter]] = {}
        for f in matches_b:
            by_name_b.setdefault(f.name.lower(), []).append(f)
        deduped_b = [_prefer_canonical(g) for g in by_name_b.values()]
        if len(deduped_b) > 1:
            table = Table(title=f"Multiple matches for '{fighter_b}'")
            table.add_column("#", style="cyan", justify="right")
            table.add_column("Name", style="bold")
            for i, f in enumerate(deduped_b, 1):
                table.add_row(str(i), f.name)
            console.print(table)
            console.print("[yellow]Multiple matches -- please be more specific[/yellow]")
            return

        fa = deduped_a[0]
        fb = deduped_b[0]

        # Same fighter check
        if fa.id == fb.id:
            console.print("[red]Cannot compare a fighter to themselves[/red]")
            return

        # Build matchup
        result = build_matchup(session, fa, fb)

        # ── Physical Comparison ──
        console.print(f"\n[bold]{result.fighter_a_name} vs {result.fighter_b_name}[/bold]\n")

        phys_table = Table(title="Physical Comparison")
        phys_table.add_column("Attribute", style="cyan")
        phys_table.add_column(result.fighter_a_name, justify="right")
        phys_table.add_column(result.fighter_b_name, justify="right")
        phys_table.add_column("Differential", justify="right")

        p = result.physical
        phys_table.add_row(
            "Height (in)",
            _format_val(getattr(fa, "height_inches", None)),
            _format_val(getattr(fb, "height_inches", None)),
            _format_diff(p.height_diff),
        )
        phys_table.add_row(
            "Reach (in)",
            _format_val(getattr(fa, "reach_inches", None)),
            _format_val(getattr(fb, "reach_inches", None)),
            _format_diff(p.reach_diff),
        )
        phys_table.add_row(
            "Leg Reach (in)",
            _format_val(getattr(fa, "leg_reach_inches", None)),
            _format_val(getattr(fb, "leg_reach_inches", None)),
            _format_diff(p.leg_reach_diff),
        )
        phys_table.add_row(
            "Age (yrs)",
            _format_val(p.fighter_a_age),
            _format_val(p.fighter_b_age),
            _format_diff(p.age_diff),
        )
        console.print(phys_table)
        console.print()

        # ── Elo Comparison ──
        elo_table = Table(title="Elo Comparison")
        elo_table.add_column("Type", style="cyan")
        elo_table.add_column(result.fighter_a_name, justify="right")
        elo_table.add_column(result.fighter_b_name, justify="right")
        elo_table.add_column("Advantage", justify="right")

        e = result.elo
        overall_diff = (
            (e.fighter_a_overall - e.fighter_b_overall)
            if e.fighter_a_overall is not None and e.fighter_b_overall is not None
            else None
        )
        striking_diff = (
            (e.fighter_a_striking - e.fighter_b_striking)
            if e.fighter_a_striking is not None and e.fighter_b_striking is not None
            else None
        )
        grappling_diff = (
            (e.fighter_a_grappling - e.fighter_b_grappling)
            if e.fighter_a_grappling is not None and e.fighter_b_grappling is not None
            else None
        )

        elo_table.add_row(
            "Overall Elo",
            _format_val(e.fighter_a_overall, ".0f"),
            _format_val(e.fighter_b_overall, ".0f"),
            _format_diff(overall_diff),
        )
        elo_table.add_row(
            "Striking Elo",
            _format_val(e.fighter_a_striking, ".0f"),
            _format_val(e.fighter_b_striking, ".0f"),
            _format_diff(striking_diff),
        )
        elo_table.add_row(
            "Grappling Elo",
            _format_val(e.fighter_a_grappling, ".0f"),
            _format_val(e.fighter_b_grappling, ".0f"),
            _format_diff(grappling_diff),
        )
        console.print(elo_table)
        console.print()

        # ── Striking Matchup ──
        str_table = Table(title="Striking Matchup")
        str_table.add_column("Metric", style="cyan")
        str_table.add_column(result.fighter_a_name, justify="right")
        str_table.add_column(result.fighter_b_name, justify="right")

        s = result.striking
        str_table.add_row(
            "Sig. Str./Min",
            _format_val(s.a_sig_str_per_min),
            _format_val(s.b_sig_str_per_min),
        )
        str_table.add_row(
            "Striking Accuracy",
            _format_pct(s.a_striking_accuracy),
            _format_pct(s.b_striking_accuracy),
        )
        str_table.add_row(
            "Strike Defense",
            _format_pct(s.a_strike_defense),
            _format_pct(s.b_strike_defense),
        )
        console.print(str_table)
        console.print()

        # ── Grappling Matchup ──
        grap_table = Table(title="Grappling Matchup")
        grap_table.add_column("Metric", style="cyan")
        grap_table.add_column(result.fighter_a_name, justify="right")
        grap_table.add_column(result.fighter_b_name, justify="right")

        g = result.grappling
        grap_table.add_row(
            "TD Rate",
            _format_val(g.a_td_rate),
            _format_val(g.b_td_rate),
        )
        grap_table.add_row(
            "TD Accuracy",
            _format_pct(g.a_td_accuracy),
            _format_pct(g.b_td_accuracy),
        )
        grap_table.add_row(
            "TD Defense",
            _format_pct(g.a_td_defense),
            _format_pct(g.b_td_defense),
        )
        grap_table.add_row(
            "Control Time (s)",
            _format_val(g.a_ctrl_time, ".0f"),
            _format_val(g.b_ctrl_time, ".0f"),
        )
        console.print(grap_table)
        console.print()

        # ── Style Analysis ──
        style_table = Table(title="Style Analysis")
        style_table.add_column("Attribute", style="cyan")
        style_table.add_column(result.fighter_a_name, justify="right")
        style_table.add_column(result.fighter_b_name, justify="right")

        st = result.style
        style_table.add_row(
            "Style Tag",
            st.fighter_a_style or "N/A",
            st.fighter_b_style or "N/A",
        )
        style_table.add_row(
            "Style Index",
            _format_val(st.fighter_a_index, ".1f"),
            _format_val(st.fighter_b_index, ".1f"),
        )
        console.print(style_table)

        # Style matchup win rates
        if st.matchup_win_rates:
            for wr in st.matchup_win_rates:
                if wr.sufficient_data:
                    console.print(
                        f"  {wr.style_a} vs {wr.style_b}: "
                        f"{wr.style_a} wins {wr.style_a_win_pct:.0%} "
                        f"({wr.total} fights)"
                    )
                else:
                    console.print(
                        f"  {wr.style_a} vs {wr.style_b}: Insufficient data (< 10 fights)"
                    )

        if st.cross_division_warning:
            console.print(f"\n[yellow]{st.cross_division_warning}[/yellow]")

        console.print()

    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise SystemExit(1) from exc
    finally:
        session.close()


# ─────────────────────────────────────────────────────────────────────────
# Phase 56 GATE-V26-03 — GATE-RECALIB-PERIODIC CLI
#
# Phase 72 METH-V261-02 disposition (2026-06-06): the v2.6.1 batch close-out
# verified that corpus-growth has NOT triggered the >10% recalibration gate
# (Phase 71 deferred prevented any data refresh; fight_odds count unchanged
# at ~25,632 since v2.6 close). The actual --apply re-derivation logic is
# therefore deferred to v2.7+ as Bucket I row METH-V27-01 with the explicit
# re-eligibility criterion "corpus growth >=10% above v2.6 close baseline of
# ~25,632 fight_odds rows". The scaffold below STAYS at its Phase 56 shape:
# dry-run reports the contract surface; --apply exits 2 with a deferred-to-
# v2.7+ message. tests/cli/test_gate_recalib_deferred.py locks this exit
# path so a future PR cannot silently flip behaviour.
# ─────────────────────────────────────────────────────────────────────────


@gate_app.command("recalib")
def gate_recalib(
    feature_set: Annotated[
        str,
        typer.Option(
            "--feature-set",
            "-f",
            help="Gate contract version to recalibrate (e.g., 'v2.6').",
        ),
    ] = "v2.6",
    apply: Annotated[
        bool,
        typer.Option(
            "--apply",
            help=(
                "Write the recalibrated contract to "
                ".planning/gate_contract_v<version>.json. Default is dry-run."
            ),
        ),
    ] = False,
    corpus_growth_threshold_pct: Annotated[
        float,
        typer.Option(
            "--threshold",
            help=(
                "Corpus-growth percentage triggering re-derivation. "
                "Default 10%% per GATE-V26-03 spec."
            ),
        ),
    ] = 10.0,
) -> None:
    """Re-derive per-slice thresholds when corpus grows above contract baseline.

    Per Phase 54 methodology spec + Phase 56 GATE-V26-03. The formula
    (D-18 LOCKED) is preserved verbatim; only the substrate measurement
    is re-run.

    Default mode is --dry-run: reports corpus-growth delta + would-be
    threshold delta WITHOUT writing a new contract file. v2.6 ships
    scaffold + dry-run; actual re-derivation logic is v2.6.1 follow-on
    (gated on the corpus-growth trigger firing).
    """
    from ufc_prediction.ml.gate_contract import load_gate_contract

    if feature_set != "v2.6":
        console.print(
            f"[red]gate recalib supports only 'v2.6' feature-set; got "
            f"{feature_set!r}.[/red] Pre-v2.6 thresholds were derived under a "
            f"non-substrate-drift–safe methodology and cannot be re-derived "
            f"without first migrating to the v2.6 verifier."
        )
        raise SystemExit(1)

    load_gate_contract.cache_clear()
    contract = load_gate_contract(version=feature_set)

    table = Table(
        title=f"GATE-RECALIB-PERIODIC — dry-run ({feature_set})",
        show_header=True,
    )
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("contract version", contract.version)
    table.add_row("derived_at", contract.derived_at)
    table.add_row(
        "formula_hash (D-18 LOCKED)",
        contract.formula_hash[:32] + "…",
    )
    table.add_row("methodology_version", contract.methodology_version)
    table.add_row(
        "substrate_alignment_strategy",
        contract.substrate_alignment_strategy,
    )
    table.add_row("confound_threshold", f"{contract.confound_threshold:.4f}")
    table.add_row(
        "corpus-growth threshold",
        f"≥{corpus_growth_threshold_pct:.1f}% above baseline triggers re-derivation",
    )
    console.print(table)
    console.print(
        "[yellow]v2.6 scope:[/yellow] CLI scaffold + dry-run report shipped. "
        "Threshold re-derivation logic is a v2.6.1 follow-on. See "
        ".planning/gate_methodology_v2.6.md §3.1 + docs/GATE_METHODOLOGY.md."
    )
    if apply:
        console.print(
            "[red]--apply requested but v2.6.1 follow-on owns the actual "
            "contract write. v2.6 is dry-run-only.[/red]"
        )
        raise SystemExit(2)
    console.print("[green]dry-run complete. No contract written.[/green]")


# ── Phase 75 METH-V27-02 — dual-substrate CLI dispatch helper (v2.7) ──────


def _gate_verify_dual_substrate(
    *,
    candidate: Path,
    canonical: Path,
    candidate_substrate: Path,
    canonical_substrate: Path,
    out: Path | None,
) -> None:
    """Phase 75 METH-V27-02 dispatch — dual-substrate verifier CLI helper.

    Extracted to a private helper so the main ``gate_verify`` function body
    keeps its Phase 55/63 single-substrate code path BYTE-IDENTICAL pre-vs-
    post Plan 75-03 (D-04 backward-compat). The helper:

      1. Loads both substrate parquets via ``load_substrate_snapshot``
         (Phase 63 R1-R8 validation; same loader used by single-substrate
         path so both code paths share validation).
      2. Invokes ``verify_candidate_vs_canonical_dual_substrate`` (Plan 75-02
         entry point) which wraps two ``verify_candidate_vs_canonical``
         calls + the D-03 LOCKED combinator.
      3. Emits via ``emit_dual_verdict_json`` (Plan 75-02 byte-stable writer).
      4. Renders two per-slice tables (Test 1 + Test 2 aligned Brier numbers)
         + a Summary table with the colored ``combined_verdict`` literal.

    Default ``--out`` path scheme: ``results/gate_verdict_v27_<candidate_stem>
    .json`` (v27 prefix distinguishes from the v26 single-substrate sidecar
    so both can coexist in ``results/`` for audit-lineage clarity).

    Contract sources:
      - Methodology spec: ``.planning/gate_methodology_v2.7.md`` §5-§6.
      - Verifier entry point: ``ufc_prediction.ml.gate_verifier
        ::verify_candidate_vs_canonical_dual_substrate`` (Plan 75-02).
    """
    from ufc_prediction.ml.gate_verifier import (
        emit_dual_verdict_json,
        verify_candidate_vs_canonical_dual_substrate,
    )
    from ufc_prediction.ml.substrate_loader import load_substrate_snapshot

    # Resolve default --out (v27 prefix; distinct from v26 sidecar).
    if out is None:
        out = Path("results") / f"gate_verdict_v27_{candidate.stem}.json"

    # Load both eval substrate slices; surface loader R1-R8 validation
    # errors as clean operator-facing messages (mirrors single-substrate
    # path's error handling).
    try:
        candidate_eval_slices = load_substrate_snapshot(candidate_substrate)
    except (ValueError, FileNotFoundError, OSError) as exc:
        console.print(f"[red]Candidate substrate load failed:[/red] {exc}")
        raise SystemExit(1) from exc
    try:
        canonical_eval_slices = load_substrate_snapshot(canonical_substrate)
    except (ValueError, FileNotFoundError, OSError) as exc:
        console.print(f"[red]Canonical substrate load failed:[/red] {exc}")
        raise SystemExit(1) from exc

    try:
        verdict = verify_candidate_vs_canonical_dual_substrate(
            candidate=candidate,
            canonical=canonical,
            candidate_eval_slices=candidate_eval_slices,
            canonical_eval_slices=canonical_eval_slices,
        )
    except ValueError as exc:
        console.print(f"[red]Dual-substrate verifier failed:[/red] {exc}")
        raise SystemExit(1) from exc

    written = emit_dual_verdict_json(verdict, out)
    console.print(f"[green]Dual-verdict written to[/green] {written}")

    # Render Test 1 + Test 2 per-slice tables.
    for test_label, sub_verdict in (
        ("Test 1 (candidate-aligned)", verdict.test_1_verdict),
        ("Test 2 (canonical-aligned)", verdict.test_2_verdict),
    ):
        table = Table(
            title=f"GATE-V27 {test_label} — {candidate.name}",
            show_header=True,
        )
        table.add_column("Slice", style="cyan")
        table.add_column("Aligned baseline", style="white")
        table.add_column("Aligned candidate", style="white")
        table.add_column("Aligned delta", style="white")
        table.add_column("Sub-verdict", style="white")
        for slice_name in sorted(sub_verdict.aligned_baseline_brier_per_slice):
            ab = sub_verdict.aligned_baseline_brier_per_slice[slice_name]
            ac = sub_verdict.aligned_candidate_brier_per_slice[slice_name]
            ad = sub_verdict.aligned_delta_per_slice[slice_name]
            table.add_row(
                slice_name,
                f"{ab:.4f}" if ab is not None else "n/a",
                f"{ac:.4f}" if ac is not None else "n/a",
                f"{ad:.4f}" if ad is not None else "n/a",
                sub_verdict.verdict,
            )
        console.print(table)

    # Summary table with colored combined_verdict literal.
    verdict_color = {
        "path_a_promote_dual": "green",
        "path_b_reject_dual": "yellow",
        "highly_suspect": "yellow",
        "substrate_drift_artifact": "red",
        "confound_block_dual": "red",
        "width_mismatch_dual": "red",
    }.get(verdict.combined_verdict, "white")
    summary = Table(title="Dual-Substrate Summary", show_header=True)
    summary.add_column("Field", style="cyan")
    summary.add_column("Value", style="white")
    summary.add_row(
        "Combined verdict",
        f"[{verdict_color}]{verdict.combined_verdict}[/{verdict_color}]",
    )
    summary.add_row("Methodology", verdict.methodology)
    summary.add_row("Verifier version", verdict.verifier_version)
    summary.add_row("Test 1 verdict", verdict.test_1_verdict.verdict)
    summary.add_row("Test 2 verdict", verdict.test_2_verdict.verdict)
    console.print(summary)
    console.print(f"[cyan]Rationale:[/cyan] {verdict.rationale}")


@gate_app.command("verify")
def gate_verify(
    candidate: Annotated[
        Path,
        typer.Option(
            "--candidate",
            "-c",
            help="Path to candidate Pipeline (.joblib).",
        ),
    ],
    substrate_parquet: Annotated[
        Path | None,
        typer.Option(
            "--substrate-parquet",
            help=(
                "Path to substrate snapshot parquet (Phase 63 schema: "
                "slice_name, feature_vector, outcome, substrate_sha). "
                "Single-substrate v2.6 methodology. Mutually exclusive "
                "with --dual-substrate (Phase 75 v2.7)."
            ),
        ),
    ] = None,
    canonical: Annotated[
        Path,
        typer.Option(
            "--canonical",
            help="Path to canonical Pipeline (e.g., models/meta/meta_v2.joblib).",
        ),
    ] = Path("models/meta/meta_v2.joblib"),
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy",
            "-s",
            help="Substrate alignment strategy: refit_baseline or dual_test_set.",
        ),
    ] = "refit_baseline",
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help=(
                "JSON sidecar output path. Default: "
                "results/gate_verdict_v26_<candidate_stem>.json for "
                "--substrate-parquet, or "
                "results/gate_verdict_v27_<candidate_stem>.json for "
                "--dual-substrate."
            ),
        ),
    ] = None,
    dual_substrate: Annotated[
        bool,
        typer.Option(
            "--dual-substrate",
            help=(
                "Enable Phase 75 METH-V27-02 dual-test methodology. "
                "Requires --candidate-substrate AND --canonical-substrate. "
                "Mutually exclusive with --substrate-parquet."
            ),
        ),
    ] = False,
    candidate_substrate: Annotated[
        Path | None,
        typer.Option(
            "--candidate-substrate",
            help=(
                "Path to candidate-aligned substrate parquet (Test 1 "
                "substrate for --dual-substrate). col[0] = candidate's OOF "
                "(e.g., xgb_v2_refv2_oof). Required when --dual-substrate "
                "is set."
            ),
        ),
    ] = None,
    canonical_substrate: Annotated[
        Path | None,
        typer.Option(
            "--canonical-substrate",
            help=(
                "Path to canonical-aligned substrate parquet (Test 2 "
                "substrate for --dual-substrate). Built by "
                "scripts/build_canonical_substrate_v27.py (Plan 75-01). "
                "col[0] = canonical training-time xgb_oof_prob. Required "
                "when --dual-substrate is set."
            ),
        ),
    ] = None,
) -> None:
    """Run the GATE-V26-02 substrate-drift–safe verifier (Phase 55).

    Loads ``--substrate-parquet`` via
    ``ufc_prediction.ml.substrate_loader.load_substrate_snapshot`` (Phase 63
    Plan 01), runs ``verify_candidate_vs_canonical`` (Phase 55), and emits
    both a Rich stdout summary table and a JSON sidecar written via
    ``emit_verdict_json``. Default sidecar path is
    ``results/gate_verdict_v26_<candidate_stem>.json`` (override with
    ``--out``).

    Contract sources:
      - Verifier contract: ``.planning/gate_methodology_v2.6.md`` §6.
      - CLI output surface: ``.planning/phases/63-substrate-snapshot-loader-crit-v261-01/63-CONTEXT.md`` §D-04.

    Phase 75 METH-V27-02 (Plan 75-03): ``--dual-substrate`` extends this
    command with the v2.7 dual-test methodology. When ``--dual-substrate``
    is set, dispatch routes to ``_gate_verify_dual_substrate`` (which wraps
    ``verify_candidate_vs_canonical_dual_substrate``). The legacy
    ``--substrate-parquet`` single-substrate path is BYTE-IDENTICAL pre-vs-
    post Plan 75-03 (D-04 backward-compat).
    """
    # Phase 75 METH-V27-02 dispatch: --dual-substrate vs --substrate-parquet
    # are mutually exclusive; exactly one of them MUST be provided. D-04
    # backward-compat: single-substrate path UNCHANGED when --dual-substrate
    # is False (legacy default).
    if dual_substrate and substrate_parquet is not None:
        raise typer.BadParameter(
            "--substrate-parquet and --dual-substrate are mutually "
            "exclusive. Use --substrate-parquet for single-substrate v2.6 "
            "methodology, OR --dual-substrate with --candidate-substrate + "
            "--canonical-substrate for v2.7 dual-substrate methodology."
        )
    if not dual_substrate and substrate_parquet is None:
        raise typer.BadParameter(
            "Either --substrate-parquet (single-substrate v2.6) or "
            "--dual-substrate (with --candidate-substrate + "
            "--canonical-substrate for v2.7) is required."
        )
    if dual_substrate and (candidate_substrate is None or canonical_substrate is None):
        raise typer.BadParameter(
            "--dual-substrate requires BOTH --candidate-substrate and "
            "--canonical-substrate to be provided."
        )

    if dual_substrate:
        # All four guard clauses above confirm both substrate paths are set.
        assert candidate_substrate is not None  # noqa: S101 — type narrowing
        assert canonical_substrate is not None  # noqa: S101 — type narrowing
        _gate_verify_dual_substrate(
            candidate=candidate,
            canonical=canonical,
            candidate_substrate=candidate_substrate,
            canonical_substrate=canonical_substrate,
            out=out,
        )
        return

    # ── Legacy single-substrate v2.6 code path (UNCHANGED from Phase 55/63)
    # The block below is BYTE-IDENTICAL to its Phase 55/63 form modulo the
    # function-signature change from `Path` → `Path | None` on the
    # ``substrate_parquet`` parameter (Plan 75-03 D-04 backward-compat
    # promotion to optional). The narrowing assertion below makes mypy-strict
    # happy without altering runtime behavior.
    assert substrate_parquet is not None  # noqa: S101 — type narrowing
    # Function-scope imports keep module-load cheap and mirror the
    # gate_recalib pattern (load_gate_contract is imported inside the
    # function body, not at module top-level).
    from ufc_prediction.ml.gate_verifier import (
        emit_verdict_json,
        verify_candidate_vs_canonical,
    )
    from ufc_prediction.ml.substrate_loader import load_substrate_snapshot

    # Resolve default --out per CONTEXT D-04. Path.stem strips the
    # `.joblib` suffix per the spec.
    if out is None:
        out = Path("results") / f"gate_verdict_v26_{candidate.stem}.json"

    # Load eval slices; surface loader R1-R8 validation errors as a
    # clean operator-facing message instead of a Python traceback.
    try:
        eval_slices = load_substrate_snapshot(substrate_parquet)
    except (ValueError, FileNotFoundError, OSError) as exc:
        console.print(f"[red]Substrate snapshot load failed:[/red] {exc}")
        raise SystemExit(1) from exc

    # Run the verifier. ``strategy`` is the existing `str` option; the
    # verifier validates it internally and raises ValueError for the
    # out-of-scope "shadow_traffic" value (Phase 55 §6.1).
    try:
        verdict = verify_candidate_vs_canonical(
            candidate=candidate,
            canonical=canonical,
            eval_slices=eval_slices,
            substrate_align_strategy=strategy,  # type: ignore[arg-type]
        )
    except ValueError as exc:
        console.print(f"[red]Verifier failed:[/red] {exc}")
        raise SystemExit(1) from exc

    # Emit the JSON sidecar via the Phase 55 byte-stable writer.
    written = emit_verdict_json(verdict, out)
    console.print(f"[green]Verdict written to[/green] {written}")

    # Per-slice metrics table.
    table = Table(
        title=f"GATE-V26-02 verdict — {candidate.name}",
        show_header=True,
    )
    table.add_column("Slice", style="cyan")
    table.add_column("Raw baseline Brier", style="white")
    table.add_column("Aligned baseline Brier", style="white")
    table.add_column("Aligned candidate Brier", style="white")
    table.add_column("Aligned delta", style="white")
    table.add_column("Floor clear", style="white")
    for slice_name in sorted(verdict.aligned_baseline_brier_per_slice):
        raw_baseline = verdict.raw_baseline_brier_per_slice[slice_name]
        aligned_baseline = verdict.aligned_baseline_brier_per_slice[slice_name]
        aligned_candidate = verdict.aligned_candidate_brier_per_slice[slice_name]
        aligned_delta = verdict.aligned_delta_per_slice[slice_name]
        floor_clear = verdict.floor_clears.get(slice_name, False)
        floor_cell = "[green]PASS[/green]" if floor_clear else "[red]FAIL[/red]"
        # Per Plan 64-01 width-mismatch guard contract, raw_* values can
        # be None (sentinel for "raw measurement structurally impossible
        # due to width mismatch"). Format defensively so the Rich table
        # render does not crash on the confound_block verdict path.
        raw_baseline_cell = "n/a" if raw_baseline is None else f"{raw_baseline:.4f}"
        table.add_row(
            slice_name,
            raw_baseline_cell,
            f"{aligned_baseline:.4f}",
            f"{aligned_candidate:.4f}",
            f"{aligned_delta:.4f}",
            floor_cell,
        )
    console.print(table)

    # Summary key/value block — verdict, confound, hurdle, SHAs.
    verdict_color = {
        "path_a_promote": "green",
        "path_b_reject": "yellow",
        "confound_block": "red",
    }.get(verdict.verdict, "white")
    summary = Table(title="Summary", show_header=True)
    summary.add_column("Field", style="cyan")
    summary.add_column("Value", style="white")
    summary.add_row(
        "Verdict",
        f"[{verdict_color}]{verdict.verdict}[/{verdict_color}]",
    )
    summary.add_row(
        "Confound detected",
        f"{verdict.confound_detected} (threshold={verdict.confound_threshold:.4f})",
    )
    summary.add_row("Hurdle value", f"{verdict.hurdle_value:.4f}")
    summary.add_row("Hurdle clears", str(verdict.hurdle_clears))
    summary.add_row("Substrate SHA", verdict.substrate_sha[:16] + "…")
    console.print(summary)
    console.print(f"[cyan]Rationale:[/cyan] {verdict.rationale}")
