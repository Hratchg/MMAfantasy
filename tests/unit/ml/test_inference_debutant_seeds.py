"""Serve-path debutant seeds: loud on an empty table, and never cache an empty result.

``_load_debutant_seeds`` feeds ``_get_latest_elo`` the Sherdog debutant seed for
overall Elo, read from the ``debutant_seed_inputs`` table. The stored substrate
(``elo_snapshots.elo_before``) was built with those seeds, so an empty (or not
yet migrated) table silently skews every debutant's serve-time
``elo_overall_diff`` toward flat 1500. That must be logged, and rows that
appear later (backfill while the API is up) must be picked up rather than
masked by a cached ``{}``.

``load_seeds_from_db`` is stubbed; the real-Postgres counterpart is
``tests/integration/elo/test_debutant_seeds_db.py``.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import psycopg
import pytest
from sqlalchemy.exc import ProgrammingError

from ufc_prediction.ml import inference_features


@pytest.fixture(autouse=True)
def _fresh_seed_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inference_features, "_debutant_seeds_cache", None, raising=False)
    monkeypatch.setattr(inference_features, "_warned_missing_seeds", False, raising=False)


@pytest.fixture
def table(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A fake seed table: ``rows`` is what the DB returns; ``reads`` counts queries."""
    state: dict[str, Any] = {"rows": {}, "reads": 0, "error": None}

    def _load(_session: Any) -> dict[int, float]:
        state["reads"] += 1
        if state["error"] is not None:
            raise state["error"]
        return dict(state["rows"])

    monkeypatch.setattr(inference_features, "load_seeds_from_db", _load)
    return state


def test_empty_table_logs_error_naming_the_table(
    table: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR, logger=inference_features.__name__):
        assert inference_features._load_debutant_seeds(MagicMock()) == {}

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert "debutant_seed_inputs" in errors[0].getMessage()


def test_empty_table_error_logged_once_per_process(
    table: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR, logger=inference_features.__name__):
        for _ in range(5):
            inference_features._load_debutant_seeds(MagicMock())

    assert len([r for r in caplog.records if r.levelno >= logging.ERROR]) == 1


def test_empty_result_not_cached_rows_added_later_are_picked_up(table: dict[str, Any]) -> None:
    assert inference_features._load_debutant_seeds(MagicMock()) == {}

    table["rows"] = {7: 1625.0}
    assert inference_features._load_debutant_seeds(MagicMock()) == {7: 1625.0}


def test_non_empty_result_is_cached(table: dict[str, Any]) -> None:
    table["rows"] = {7: 1625.0}

    first = inference_features._load_debutant_seeds(MagicMock())
    table["rows"] = {}
    assert inference_features._load_debutant_seeds(MagicMock()) is first
    assert table["reads"] == 1


def test_missing_table_serves_unseeded_with_one_error(
    table: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    table["error"] = ProgrammingError(
        "SELECT", {}, psycopg.errors.UndefinedTable("relation does not exist")
    )

    with caplog.at_level(logging.ERROR, logger=inference_features.__name__):
        assert inference_features._load_debutant_seeds(MagicMock()) == {}
        assert inference_features._load_debutant_seeds(MagicMock()) == {}

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert "alembic upgrade head" in errors[0].getMessage()


def test_seed_query_runs_in_a_savepoint(table: dict[str, Any]) -> None:
    """A failing seed query must not abort the request's outer transaction."""
    session = MagicMock()
    inference_features._load_debutant_seeds(session)
    session.begin_nested.assert_called_once()


def test_no_csv_path_is_consulted() -> None:
    """The DB is the single source of truth: no CSV fallback on the serve path."""
    assert not hasattr(inference_features, "_SHERDOG_PRE_UFC_CSV")
    assert not hasattr(inference_features, "load_seeds")
