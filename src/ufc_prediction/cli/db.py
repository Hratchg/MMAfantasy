from __future__ import annotations

import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import psycopg
import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session as SASession

from ufc_prediction.config import settings

DEFAULT_DUMP_PATH = Path("data/seed/ufc_corpus_v30.dump")

CANONICAL_TABLES = (
    "events",
    "fights",
    "fighters",
    "fighter_aliases",
    "fight_odds",
    "round_stats",
    "elo_snapshots",
    "computed_features",
    "referees",
    "venues",
    "model_runs",
    "alembic_version",
)

db_app = typer.Typer(name="db", help="Database seed and status commands")
console = Console()


def _normalize_for_psycopg(url: str) -> str:
    # SQLAlchemy-style URLs like postgresql+psycopg://... must drop the driver
    # tag before psycopg.connect() will parse them.
    if url.startswith("postgresql+psycopg://"):
        return "postgresql://" + url[len("postgresql+psycopg://") :]
    return url


def _sqlalchemy_url(url: str) -> str:
    # create_engine needs an explicit driver; a bare postgresql:// defaults to
    # psycopg2 (not installed → ModuleNotFoundError) and postgres:// is rejected
    # outright. Normalize both to the +psycopg form so _session_for accepts every
    # URL shape _check_reachable (via _normalize_for_psycopg + psycopg.connect)
    # already tolerates — review #12.
    if url.startswith("postgresql+"):
        return url
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    return url


@contextmanager
def _session_for(url: str):
    """Yield a Session bound to the RUNTIME-resolved ``url``.

    #8 (review): ``seed()``/``status()`` resolve ``DATABASE_URL`` at call time,
    but the emptiness gate and row counts previously went through the module-
    level ``SessionLocal`` whose engine is bound once at import. If the env is
    mutated after import (test harness / in-process wrapper), the checks would
    inspect one DB while ``pg_restore --clean`` clobbers another. Binding a
    fresh engine to the resolved URL keeps inspection and restore consistent.
    """
    engine = create_engine(_sqlalchemy_url(url))
    try:
        with SASession(engine) as session:
            yield session
    finally:
        engine.dispose()


def _check_database_url() -> str:
    url = os.getenv("DATABASE_URL") or settings.database_url
    if not url:
        console.print(
            "[red]Pre-flight 1/5 FAILED:[/red] DATABASE_URL is not set (see docs/INSTALL.md step 3)"
        )
        raise typer.Exit(1)
    return url


def _check_reachable(url: str) -> None:
    try:
        conn = psycopg.connect(_normalize_for_psycopg(url), connect_timeout=5)
        conn.close()
    except Exception as exc:
        console.print(
            f"[red]Pre-flight 2/5 FAILED:[/red] Postgres not reachable at "
            f"{url}: {exc}; verify Docker container is running"
        )
        raise typer.Exit(1) from exc


def _check_pg_restore() -> str:
    path = shutil.which("pg_restore")
    if path is None:
        console.print(
            "[red]Pre-flight 3/5 FAILED:[/red] pg_restore not on PATH "
            "(install via `brew install postgresql@16` or use Docker)"
        )
        raise typer.Exit(1)
    return path


def _check_source(from_: Path) -> None:
    if not from_.exists():
        console.print(
            f"[red]Pre-flight 4/5 FAILED:[/red] dump file not found: {from_} "
            "(pass --from to override the default path)"
        )
        raise typer.Exit(1)


def _check_target_empty(force: bool, url: str) -> None:
    with _session_for(url) as session:
        non_empty: list[tuple[str, int]] = []
        for table in CANONICAL_TABLES:
            if table == "alembic_version":
                continue
            try:
                count = session.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar() or 0
            except ProgrammingError as exc:
                # A statement error aborts the tx in Postgres — roll back so the
                # remaining COUNTs still run.
                session.rollback()
                if isinstance(getattr(exc, "orig", None), psycopg.errors.UndefinedTable):
                    # Table absent on a fresh/partial DB → legitimately empty.
                    continue
                # #4 (review): FAIL CLOSED. Any error OTHER than "table does not
                # exist" must NOT be silently read as empty — that would let a
                # transient permission/lock/timeout error greenlight a
                # destructive `pg_restore --clean` over a populated DB.
                console.print(
                    f"[red]Pre-flight 5/5 FAILED:[/red] could not read table "
                    f"{table!r}: {exc}. Refusing to treat as empty; aborting "
                    "(re-run once the DB is healthy, or drop it manually)."
                )
                raise typer.Exit(1) from exc
            except Exception as exc:
                session.rollback()
                console.print(
                    f"[red]Pre-flight 5/5 FAILED:[/red] error counting {table!r}: "
                    f"{exc}. Refusing to treat as empty; aborting."
                )
                raise typer.Exit(1) from exc
            if count > 0:
                non_empty.append((table, int(count)))
        if non_empty and not force:
            listing = ", ".join(f"{t}={n}" for t, n in non_empty)
            console.print(
                f"[red]Pre-flight 5/5 FAILED:[/red] DB has data in tables: "
                f"{listing}; use --force to drop+restore"
            )
            raise typer.Exit(1)


def _row_counts(url: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    with _session_for(url) as session:
        for table in CANONICAL_TABLES:
            try:
                value = session.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            except ProgrammingError as exc:
                session.rollback()
                if isinstance(getattr(exc, "orig", None), psycopg.errors.UndefinedTable):
                    value = 0  # absent table → 0 (display only, post-restore/status)
                else:
                    raise
            counts[table] = int(value or 0)
    return counts


def _print_row_table(counts: dict[str, int], title: str) -> None:
    table = Table(title=title)
    table.add_column("Table", style="cyan")
    table.add_column("Row count", justify="right", style="green")
    for name in CANONICAL_TABLES:
        table.add_row(name, f"{counts.get(name, 0):,}")
    console.print(table)


def _alembic_stamp_head() -> None:
    result = subprocess.run(
        ["alembic", "stamp", "head"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        console.print(f"[red]alembic stamp head FAILED:[/red] {result.stderr}")
        raise typer.Exit(1)


def _predictor_sanity_check() -> None:
    from ufc_prediction.ml.predictor import ModelPredictor

    try:
        ModelPredictor(model_dir="models", version="v2")
    except Exception as exc:
        console.print(
            "[red]Predictor sanity check FAILED[/red] — corpus restored but "
            "ModelPredictor(version='v2') cannot instantiate. This is the "
            "predictor-default-version regression documented in KNOWN_ISSUES.md."
        )
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    console.print(
        "[green]✓ ModelPredictor v2 instantiated:[/green] META-V22 canonical (xgb_v2 + meta_v2)"
    )


@db_app.command("seed")
def seed(
    from_: Path = typer.Option(
        DEFAULT_DUMP_PATH,
        "--from",
        help="Path to pg_dump custom-format file (default: data/seed/ufc_corpus_v30.dump)",
    ),
    force: bool = typer.Option(False, "--force", help="Restore over non-empty target DB"),
    no_migrate: bool = typer.Option(
        False, "--no-migrate", help="Skip `alembic stamp head` after restore"
    ),
) -> None:
    """Restore the corpus snapshot into the local Postgres."""
    t0 = time.monotonic()
    url = _check_database_url()
    _check_reachable(url)
    pg_restore = _check_pg_restore()
    _check_source(from_)
    _check_target_empty(force, url)

    result = subprocess.run(
        [
            pg_restore,
            "--no-owner",
            "--no-privileges",
            "--clean",
            "--if-exists",
            f"--dbname={_normalize_for_psycopg(url)}",
            str(from_),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        console.print(f"[red]pg_restore FAILED:[/red] {result.stderr}")
        raise typer.Exit(1)

    if not no_migrate:
        _alembic_stamp_head()

    _print_row_table(_row_counts(url), title="ufc db seed — row counts")
    _predictor_sanity_check()

    elapsed = time.monotonic() - t0
    console.print(f"[green]✓ ufc db seed complete[/green] ({elapsed:.1f}s)")


@db_app.command("status")
def status() -> None:
    """Report row counts per table + alembic head version."""
    url = _check_database_url()
    _check_reachable(url)
    _print_row_table(_row_counts(url), title="ufc db status")
    with _session_for(url) as session:
        try:
            head = session.execute(text("SELECT version_num FROM alembic_version")).scalar()
        except Exception:
            head = None
    console.print(f"alembic head: [cyan]{head or '(unknown)'}[/cyan]")
