"""`ufc elo compute` fails closed when the DB holds no debutant seeds.

The seeds live in the ``debutant_seed_inputs`` table. Without them the engine
falls back to flat-1500 debutant ratings, and ``flush_snapshots`` would then
delete and rewrite every overall ``elo_snapshots`` row in the shared DB with
that degraded substrate. The command must refuse before loading fights or
writing anything unless the operator passes ``--allow-unseeded`` for an
intentional flat-1500 run.

No DB and no real ``elo compute``: every DB touchpoint is monkeypatched. The
real-Postgres counterpart is ``tests/integration/elo/test_debutant_seeds_db.py``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import psycopg
import pytest
from sqlalchemy.exc import OperationalError, ProgrammingError
from typer.testing import CliRunner

from ufc_prediction.cli import main as cli_main
from ufc_prediction.cli.main import app

runner = CliRunner()


def _fake_db(monkeypatch: pytest.MonkeyPatch, seeds: Any) -> dict[str, int]:
    """Stub every DB touchpoint of compute_elo and count the ones that matter.

    ``seeds`` is what ``load_seeds_from_db`` returns, or an exception it raises.
    """
    calls = {"fights": 0, "flush": 0}

    def _load(_s: Any) -> dict[int, float]:
        if isinstance(seeds, BaseException):
            raise seeds
        return dict(seeds)

    def _fights(_s: Any) -> list[Any]:
        calls["fights"] += 1
        return []

    def _flush(*_a: Any, **_k: Any) -> int:
        calls["flush"] += 1
        return 0

    monkeypatch.setattr(cli_main, "SessionLocal", lambda: MagicMock())
    monkeypatch.setattr(cli_main, "load_seeds_from_db", _load)
    monkeypatch.setattr(cli_main, "load_fights_chronological", _fights)
    monkeypatch.setattr(cli_main, "flush_snapshots", _flush)
    monkeypatch.setattr(cli_main, "flush_domain_snapshots", _flush)
    return calls


def _undefined_table() -> ProgrammingError:
    return ProgrammingError(
        "SELECT ... FROM debutant_seed_inputs",
        {},
        psycopg.errors.UndefinedTable('relation "debutant_seed_inputs" does not exist'),
    )


def test_empty_seed_table_exits_1_before_any_write(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_db(monkeypatch, {})

    result = runner.invoke(app, ["elo", "compute", "--no-domain"])

    assert result.exit_code == 1, result.output
    assert "--allow-unseeded" in result.output
    assert "backfill-pre-ufc-seeds" in result.output
    assert calls == {"fights": 0, "flush": 0}


def test_missing_seed_table_exits_1_and_names_the_migration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_db(monkeypatch, _undefined_table())

    result = runner.invoke(app, ["elo", "compute", "--no-domain"])

    assert result.exit_code == 1, result.output
    assert "alembic upgrade head" in result.output
    assert calls == {"fights": 0, "flush": 0}


def test_other_db_errors_reading_seeds_exit_1_without_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_db(monkeypatch, OperationalError("SELECT", {}, Exception("server closed")))

    result = runner.invoke(app, ["elo", "compute", "--no-domain", "--allow-unseeded"])

    assert result.exit_code == 1, result.output
    assert calls == {"fights": 0, "flush": 0}


def test_allow_unseeded_runs_flat_1500_path(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_db(monkeypatch, {})

    result = runner.invoke(app, ["elo", "compute", "--no-domain", "--allow-unseeded"])

    assert result.exit_code == 0, result.output
    assert calls == {"fights": 1, "flush": 1}


def test_allow_unseeded_also_covers_a_missing_seed_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_db(monkeypatch, _undefined_table())

    result = runner.invoke(app, ["elo", "compute", "--no-domain", "--allow-unseeded"])

    assert result.exit_code == 0, result.output
    assert calls == {"fights": 1, "flush": 1}


def test_seeded_table_runs_normally(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_db(monkeypatch, {1: 1600.0})

    result = runner.invoke(app, ["elo", "compute", "--no-domain"])

    assert result.exit_code == 0, result.output
    assert "Loaded 1 debutant Elo seeds from debutant_seed_inputs" in result.output
    assert calls == {"fights": 1, "flush": 1}


def test_no_csv_path_is_consulted(monkeypatch: pytest.MonkeyPatch) -> None:
    """The DB is the single source of truth: no CSV fallback on this path."""
    assert not hasattr(cli_main, "_SHERDOG_PRE_UFC_CSV")
    assert not hasattr(cli_main, "load_seeds")
