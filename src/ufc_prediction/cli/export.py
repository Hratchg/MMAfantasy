"""CLI export commands for fighter data, rankings, and Elo history."""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from ufc_prediction.db.session import SessionLocal
from ufc_prediction.elo.fighter_queries import (
    get_all_fighters_with_ratings,
    get_division_rankings,
    get_elo_history,
    list_divisions,
    resolve_weight_class,
    search_fighters,
)

export_app = typer.Typer(name="export", help="Export data to CSV/JSON")


def _write_csv(data: list[dict[str, Any]], output: Path | None) -> None:
    """Write list of dicts as CSV to stdout or file per D-10."""
    if not data:
        typer.echo("No data to export.")
        return
    fieldnames = list(data[0].keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in data:
        cleaned = {k: ("" if v is None else v) for k, v in row.items()}
        writer.writerow(cleaned)
    content = buf.getvalue()
    if output is not None:
        output.write_text(content, encoding="utf-8")
        typer.echo(f"Exported to {output}")
    else:
        sys.stdout.write(content)


def _write_json(data: list[dict[str, Any]], output: Path | None) -> None:
    """Write list of dicts as JSON to stdout or file per D-10."""
    if not data:
        typer.echo("No data to export.")
        return

    def default_serializer(obj: Any) -> str:
        """Handle date/datetime serialization."""
        if hasattr(obj, "isoformat"):
            return str(obj.isoformat())
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    content = json.dumps(data, indent=2, default=default_serializer)
    if output is not None:
        output.write_text(content, encoding="utf-8")
        typer.echo(f"Exported to {output}")
    else:
        sys.stdout.write(content + "\n")


def _write_output(data: list[dict[str, Any]], fmt: str, output: Path | None) -> None:
    """Route to CSV or JSON writer."""
    if fmt == "json":
        _write_json(data, output)
    else:
        _write_csv(data, output)


@export_app.command("fighters")
def export_fighters(
    format: Annotated[str, typer.Option("--format", help="Output format: csv or json")] = "csv",
    output: Annotated[Path | None, typer.Option(help="Output file (default: stdout)")] = None,
) -> None:
    """Export all fighter profiles with Elo ratings (per D-09, D-11)."""
    session = SessionLocal()
    try:
        data = get_all_fighters_with_ratings(session)
        _write_output(data, format, output)
    finally:
        session.close()


@export_app.command("rankings")
def export_rankings(
    division: Annotated[str, typer.Argument(help="Weight class (e.g., 'Lightweight')")],
    format: Annotated[str, typer.Option("--format", help="Output format: csv or json")] = "csv",
    output: Annotated[Path | None, typer.Option(help="Output file (default: stdout)")] = None,
    top: Annotated[int, typer.Option(help="Number of fighters")] = 50,
) -> None:
    """Export division rankings (per D-09, D-11)."""
    matches = resolve_weight_class(division)
    if not matches:
        typer.echo(f"Division '{division}' not found. Valid: {', '.join(sorted(list_divisions()))}")
        raise SystemExit(1)
    if len(matches) > 1:
        typer.echo(f"Ambiguous division '{division}'. Matches: {', '.join(matches)}")
        raise SystemExit(1)

    resolved = matches[0]
    session = SessionLocal()
    try:
        data = get_division_rankings(session, resolved, limit=top)
        # Add rank field
        for i, row in enumerate(data, 1):
            row["rank"] = i
        _write_output(data, format, output)
    finally:
        session.close()


@export_app.command("history")
def export_history(
    fighter_name: Annotated[str, typer.Argument(help="Fighter name to search for")],
    format: Annotated[str, typer.Option("--format", help="Output format: csv or json")] = "csv",
    output: Annotated[Path | None, typer.Option(help="Output file (default: stdout)")] = None,
) -> None:
    """Export a fighter's Elo history over time (per D-11)."""
    session = SessionLocal()
    try:
        matches = search_fighters(session, fighter_name)
        if not matches:
            typer.echo(f"Fighter '{fighter_name}' not found.")
            raise SystemExit(1)
        if len(matches) > 1:
            typer.echo(f"Multiple matches for '{fighter_name}':")
            for f in matches:
                typer.echo(f"  - {f.name}")
            typer.echo("Please be more specific.")
            raise SystemExit(1)

        fighter = matches[0]
        data = get_elo_history(session, fighter.id)
        # Add fighter name to each row
        for row in data:
            row["fighter_name"] = fighter.name
        _write_output(data, format, output)
    finally:
        session.close()
