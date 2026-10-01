"""Sherdog debutant seeds read from the ``debutant_seed_inputs`` table (testcontainers).

The seeds used to come only from the untracked ``data/sherdog/pre_ufc_records.csv``,
which the Docker serving image does not ship. They now live in the DB:

- ``ufc db backfill-pre-ufc-seeds`` loads the CSV idempotently;
- ``elo compute`` and the serve path (``ml.inference_features``) read the table.

The load-bearing property is equivalence: the seeds derived from the DB rows
are the same dict, with the same floats, as ``load_seeds(csv)`` over the same
records, so moving the source changes no rating.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from ufc_prediction.db.base import Base
from ufc_prediction.elo.seed import load_seeds
from ufc_prediction.elo.seed_store import (
    load_seeds_from_db,
    parse_seed_csv,
    seeded_fighter_ids,
    unknown_fighter_ids,
    upsert_seed_rows,
)


def _docker_available() -> bool:
    try:
        import docker  # type: ignore[import-untyped]

        docker.from_env().ping()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _docker_available(), reason="Docker daemon unavailable"),
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REAL_CSV = _REPO_ROOT / "data" / "sherdog" / "pre_ufc_records.csv"

_HEADER = [
    "fighter_id",
    "sherdog_url",
    "n_pre_ufc_fights",
    "wins",
    "losses",
    "draws",
    "nc_dq",
    "win_rate",
    "kos",
    "submissions",
    "decisions",
    "last_organization",
    "org_tier",
    "scraped_at",
]

# Fighter ids far above the real corpus so committed rows never collide.
_BASE_ID = 910_000

# Edge cases for exact float round-tripping (long reprs, 0.1+0.2, 0/1 bounds)
# plus every tier, a zero-fight row and a >20-fight row (experience cap).
_ROWS: list[dict[str, str]] = [
    {"n": "9", "win_rate": "0.7777777777777778", "org_tier": "local"},
    {"n": "3", "win_rate": "0.3333333333333333", "org_tier": "regional"},
    {"n": "3", "win_rate": "0.6666666666666666", "org_tier": "major"},
    {"n": "10", "win_rate": "0.30000000000000004", "org_tier": "local"},
    {"n": "1", "win_rate": "1.0", "org_tier": "local"},
    {"n": "40", "win_rate": "0.925", "org_tier": "major"},
    {"n": "0", "win_rate": "0.0", "org_tier": "none"},
    {"n": "17", "win_rate": "0.8823529411764706", "org_tier": "regional"},
]


def _csv_rows(win_rate_override: dict[int, str] | None = None) -> list[dict[str, str]]:
    out = []
    for i, spec in enumerate(_ROWS):
        fid = _BASE_ID + i
        win_rate = (win_rate_override or {}).get(fid, spec["win_rate"])
        out.append(
            {
                "fighter_id": str(fid),
                "sherdog_url": f"https://www.sherdog.com/fighter/Seed-Fixture-{fid}",
                "n_pre_ufc_fights": spec["n"],
                "wins": "",
                "losses": "",
                "draws": "0",
                "nc_dq": "0",
                "win_rate": win_rate,
                "kos": "0",
                "submissions": "0",
                "decisions": "0",
                "last_organization": "" if spec["org_tier"] == "none" else "Some Promotion 12",
                "org_tier": spec["org_tier"],
                "scraped_at": "2026-07-03T23:38:16.314549+00:00",
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _insert_fighters(conn: Any, ids: list[int]) -> None:
    conn.execute(
        text(
            "INSERT INTO fighters (id, name, source, created_at, updated_at) "
            "VALUES (:id, :name, 'ufcstats', now(), now()) ON CONFLICT (id) DO NOTHING"
        ),
        [{"id": fid, "name": f"Seed Fixture {fid}"} for fid in ids],
    )


@pytest.fixture
def schema(engine: Engine) -> Engine:
    """The full ORM schema (idempotent: earlier migration tests may have reset it)."""
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def tx_session(schema: Engine) -> Iterator[Session]:
    """A session whose commits are savepoints inside one rolled-back transaction."""
    connection = schema.connect()
    outer = connection.begin()
    sess = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        yield sess
    finally:
        sess.close()
        outer.rollback()
        connection.close()


@pytest.fixture
def committed_fighters(schema: Engine) -> Iterator[list[int]]:
    """Fixture fighters committed for real (the backfill CLI opens its own engine)."""
    ids = [_BASE_ID + i for i in range(len(_ROWS))]
    with schema.begin() as conn:
        _insert_fighters(conn, ids)
    try:
        yield ids
    finally:
        with schema.begin() as conn:
            conn.execute(
                text("DELETE FROM debutant_seed_inputs WHERE fighter_id >= :lo"), {"lo": _BASE_ID}
            )
            conn.execute(text("DELETE FROM fighters WHERE id >= :lo"), {"lo": _BASE_ID})


# ── Equivalence (the seeds do not change when the source moves) ────────────


def test_db_seeds_equal_load_seeds_csv_for_the_same_records(
    tx_session: Session, tmp_path: Path
) -> None:
    csv_path = _write_csv(tmp_path / "pre_ufc_records.csv", _csv_rows())
    rows = parse_seed_csv(csv_path)
    _insert_fighters(tx_session.connection(), [r["fighter_id"] for r in rows])

    upsert_seed_rows(tx_session, rows)
    from_db = load_seeds_from_db(tx_session)
    from_csv = load_seeds(csv_path)

    assert from_db == from_csv
    assert len(from_db) == len(_ROWS)
    # Same floats bit for bit, not merely approximately.
    assert {k: v.hex() for k, v in from_db.items()} == {k: v.hex() for k, v in from_csv.items()}


@pytest.mark.skipif(not _REAL_CSV.exists(), reason=f"{_REAL_CSV} not present (untracked)")
def test_db_seeds_equal_load_seeds_for_the_real_sherdog_csv(tx_session: Session) -> None:
    rows = parse_seed_csv(_REAL_CSV)
    _insert_fighters(tx_session.connection(), [r["fighter_id"] for r in rows])

    result = upsert_seed_rows(tx_session, rows)
    from_db = load_seeds_from_db(tx_session)
    from_csv = load_seeds(_REAL_CSV)

    assert result.inserted == len(rows)
    assert from_db == from_csv
    assert {k: v.hex() for k, v in from_db.items()} == {k: v.hex() for k, v in from_csv.items()}


# ── Upsert semantics ───────────────────────────────────────────────────────


def test_upsert_is_idempotent_and_rewrites_only_changed_rows(
    tx_session: Session, tmp_path: Path
) -> None:
    rows = parse_seed_csv(_write_csv(tmp_path / "a.csv", _csv_rows()))
    _insert_fighters(tx_session.connection(), [r["fighter_id"] for r in rows])

    first = upsert_seed_rows(tx_session, rows)
    again = upsert_seed_rows(tx_session, rows)
    changed = parse_seed_csv(
        _write_csv(tmp_path / "b.csv", _csv_rows(win_rate_override={_BASE_ID: "0.5"}))
    )
    third = upsert_seed_rows(tx_session, changed)

    n = len(_ROWS)
    assert (first.inserted, first.updated, first.unchanged) == (n, 0, 0)
    assert (again.inserted, again.updated, again.unchanged) == (0, 0, n)
    assert (third.inserted, third.updated, third.unchanged) == (0, 1, n - 1)
    assert load_seeds_from_db(tx_session)[_BASE_ID] == 1500.0 + 0.0 - 25.0 + 11.25
    assert seeded_fighter_ids(tx_session) >= {r["fighter_id"] for r in rows}


def test_unknown_fighter_ids_reports_ids_missing_from_fighters(tx_session: Session) -> None:
    _insert_fighters(tx_session.connection(), [_BASE_ID])
    assert unknown_fighter_ids(tx_session, [_BASE_ID, _BASE_ID + 99]) == {_BASE_ID + 99}


def test_load_seeds_from_db_empty_table_is_empty_dict(tx_session: Session) -> None:
    tx_session.execute(text("DELETE FROM debutant_seed_inputs"))
    assert load_seeds_from_db(tx_session) == {}


# ── `ufc db backfill-pre-ufc-seeds` ────────────────────────────────────────


def _backfill(url: str, *args: str) -> Any:
    from ufc_prediction.cli.main import app

    return CliRunner().invoke(
        app, ["db", "backfill-pre-ufc-seeds", *args], env={"DATABASE_URL": url}
    )


def _table_seeds(engine: Engine) -> dict[int, float]:
    with Session(engine) as sess:
        return {k: v for k, v in load_seeds_from_db(sess).items() if k >= _BASE_ID}


def test_backfill_cli_loads_csv_and_is_idempotent(
    postgres_container: Any, schema: Engine, committed_fighters: list[int], tmp_path: Path
) -> None:
    url = postgres_container.get_connection_url()
    csv_path = _write_csv(tmp_path / "pre_ufc_records.csv", _csv_rows())
    n = len(_ROWS)

    first = _backfill(url, "--csv", str(csv_path))
    assert first.exit_code == 0, first.output
    assert f"{n} inserted, 0 updated, 0 unchanged" in first.output
    assert _table_seeds(schema) == load_seeds(csv_path)

    second = _backfill(url, "--csv", str(csv_path))
    assert second.exit_code == 0, second.output
    assert f"0 inserted, 0 updated, {n} unchanged" in second.output
    assert _table_seeds(schema) == load_seeds(csv_path)


def test_backfill_cli_dry_run_writes_nothing(
    postgres_container: Any, schema: Engine, committed_fighters: list[int], tmp_path: Path
) -> None:
    url = postgres_container.get_connection_url()
    csv_path = _write_csv(tmp_path / "pre_ufc_records.csv", _csv_rows())

    result = _backfill(url, "--csv", str(csv_path), "--dry-run")

    assert result.exit_code == 0, result.output
    assert "dry run" in result.output.lower()
    assert _table_seeds(schema) == {}


def test_backfill_cli_unknown_fighter_exits_1_before_writing(
    postgres_container: Any, schema: Engine, committed_fighters: list[int], tmp_path: Path
) -> None:
    url = postgres_container.get_connection_url()
    rows = _csv_rows()
    rows.append({**rows[0], "fighter_id": str(_BASE_ID + 500)})
    csv_path = _write_csv(tmp_path / "pre_ufc_records.csv", rows)

    result = _backfill(url, "--csv", str(csv_path))

    assert result.exit_code == 1, result.output
    assert str(_BASE_ID + 500) in result.output
    assert _table_seeds(schema) == {}


def test_backfill_cli_malformed_csv_exits_1_before_writing(
    postgres_container: Any, schema: Engine, committed_fighters: list[int], tmp_path: Path
) -> None:
    url = postgres_container.get_connection_url()
    rows = _csv_rows()
    rows[3]["org_tier"] = "elite"
    csv_path = _write_csv(tmp_path / "pre_ufc_records.csv", rows)

    result = _backfill(url, "--csv", str(csv_path))

    assert result.exit_code == 1, result.output
    assert _table_seeds(schema) == {}


# ── `ufc elo compute` reads the table and fails closed on it ───────────────


@pytest.fixture
def elo_cli(
    tx_session: Session, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Callable[..., Any], dict[str, int]]]:
    """Run ``elo compute`` on the real test DB with no fight loading or snapshot writes."""
    from ufc_prediction.cli import main as cli_main

    calls = {"fights": 0, "flush": 0}

    def _fights(_s: Any) -> list[Any]:
        calls["fights"] += 1
        return []

    def _flush(*_a: Any, **_k: Any) -> int:
        calls["flush"] += 1
        return 0

    monkeypatch.setattr(cli_main, "SessionLocal", lambda: tx_session)
    monkeypatch.setattr(cli_main, "load_fights_chronological", _fights)
    monkeypatch.setattr(cli_main, "flush_snapshots", _flush)

    def _run(*args: str) -> Any:
        return CliRunner().invoke(cli_main.app, ["elo", "compute", "--no-domain", *args])

    yield _run, calls


def test_elo_compute_empty_seed_table_exits_1_before_any_write(
    tx_session: Session, elo_cli: tuple[Callable[..., Any], dict[str, int]]
) -> None:
    run, calls = elo_cli
    tx_session.execute(text("DELETE FROM debutant_seed_inputs"))

    result = run()

    assert result.exit_code == 1, result.output
    assert "--allow-unseeded" in result.output
    assert calls == {"fights": 0, "flush": 0}


def test_elo_compute_missing_seed_table_exits_1_and_names_the_migration(
    tx_session: Session, elo_cli: tuple[Callable[..., Any], dict[str, int]]
) -> None:
    run, calls = elo_cli
    tx_session.execute(text("DROP TABLE debutant_seed_inputs"))
    tx_session.commit()

    result = run()

    assert result.exit_code == 1, result.output
    assert "alembic upgrade head" in result.output
    assert calls == {"fights": 0, "flush": 0}


def test_elo_compute_reads_seeds_from_the_table(
    tx_session: Session, elo_cli: tuple[Callable[..., Any], dict[str, int]], tmp_path: Path
) -> None:
    run, calls = elo_cli
    tx_session.execute(text("DELETE FROM debutant_seed_inputs"))
    rows = parse_seed_csv(_write_csv(tmp_path / "pre_ufc_records.csv", _csv_rows()))
    _insert_fighters(tx_session.connection(), [r["fighter_id"] for r in rows])
    upsert_seed_rows(tx_session, rows)
    tx_session.commit()

    result = run()

    assert result.exit_code == 0, result.output
    assert f"Loaded {len(_ROWS)} debutant Elo seeds from debutant_seed_inputs" in result.output
    assert calls == {"fights": 1, "flush": 1}


# ── Serve path (`ml.inference_features._load_debutant_seeds`) ──────────────


@pytest.fixture
def serve_seeds(monkeypatch: pytest.MonkeyPatch) -> Any:
    from ufc_prediction.ml import inference_features

    monkeypatch.setattr(inference_features, "_debutant_seeds_cache", None)
    monkeypatch.setattr(inference_features, "_warned_missing_seeds", False)
    return inference_features


def test_serve_path_reads_the_same_seeds_as_the_csv(
    tx_session: Session, serve_seeds: Any, tmp_path: Path
) -> None:
    tx_session.execute(text("DELETE FROM debutant_seed_inputs"))
    csv_path = _write_csv(tmp_path / "pre_ufc_records.csv", _csv_rows())
    rows = parse_seed_csv(csv_path)
    _insert_fighters(tx_session.connection(), [r["fighter_id"] for r in rows])
    upsert_seed_rows(tx_session, rows)

    assert serve_seeds._load_debutant_seeds(tx_session) == load_seeds(csv_path)


def test_serve_path_missing_table_logs_error_and_keeps_the_session_usable(
    tx_session: Session, serve_seeds: Any, caplog: pytest.LogCaptureFixture
) -> None:
    tx_session.execute(text("DROP TABLE debutant_seed_inputs"))

    with caplog.at_level(logging.ERROR, logger=serve_seeds.__name__):
        assert serve_seeds._load_debutant_seeds(tx_session) == {}

    assert len([r for r in caplog.records if r.levelno >= logging.ERROR]) == 1
    # The failed seed query must not poison the request's transaction.
    assert tx_session.execute(text("SELECT 1")).scalar() == 1


# ── Sherdog ingest (scripts/ingest_pre_ufc_records_v25.py) writes the DB ───


def test_ingest_script_db_write_matches_its_csv_export(tx_session: Session, tmp_path: Path) -> None:
    """The rows the scrape persists to the DB seed exactly what its CSV export seeds."""
    import sys

    scripts_dir = str(_REPO_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import ingest_pre_ufc_records_v25 as ingest

    from ufc_prediction.scraper.sherdog_models import PreUFCRecord

    tx_session.execute(text("DELETE FROM debutant_seed_inputs"))
    rows = []
    for i, (wins, losses, tier) in enumerate([(7, 2, "local"), (11, 0, "major"), (0, 0, "none")]):
        total = wins + losses
        record = PreUFCRecord(
            total_wins=wins,
            total_losses=losses,
            total_draws=0,
            total_fights=total,
            win_pct=wins / total if total else 0.0,
            ko_finish_rate=0.0,
            sub_finish_rate=0.0,
            decision_rate=1.0 if wins else 0.0,
            career_years=1.0,
            fights=[],
        )
        rows.append(
            ingest.build_csv_row(
                fighter_id=_BASE_ID + i,
                sherdog_url=f"https://www.sherdog.com/fighter/Ingest-{i}",
                record=record,
                last_org=None if tier == "none" else "Some Org",
                tier=tier,
                scraped_at="2026-07-03T23:38:16.314549+00:00",
            )
        )
    _insert_fighters(tx_session.connection(), [int(r["fighter_id"]) for r in rows])

    result = ingest.upsert_db(rows, session_factory=lambda: tx_session)
    export = tmp_path / "pre_ufc_records.csv"
    ingest.upsert_csv(export, rows)

    assert result is not None and result.inserted == 3
    assert load_seeds_from_db(tx_session) == load_seeds(export)
